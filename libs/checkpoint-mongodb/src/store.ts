import { type MongoClient, type Db as MongoDatabase } from "mongodb";
import {
  BaseStore,
  type Operation,
  type OperationResults,
  type Item,
  type PutOperation,
  type GetOperation,
  type ListNamespacesOperation,
  type SearchOperation,
  type SearchItem,
  InvalidNamespaceError,
} from "@langchain/langgraph-checkpoint";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

/**
 * Computes the namespace path prefixes for denormalization.
 * E.g., ["a", "b", "c"] → ["a", "a/b", "a/b/c"]
 * Used for Atlas vector search pre-filtering.
 */
function computeNamespacePath(namespace: string[]): string[] {
  const paths: string[] = [];
  for (let i = 1; i <= namespace.length; i++) {
    paths.push(namespace.slice(0, i).join("/"));
  }
  return paths;
}

/**
 * Validates the provided namespace.
 * @param namespace The namespace to validate.
 * @throws {InvalidNamespaceError} If the namespace is invalid.
 */
function validateNamespace(namespace: string[]): void {
  if (namespace.length === 0) {
    throw new InvalidNamespaceError("Namespace cannot be empty.");
  }
  for (const label of namespace) {
    if (typeof label !== "string") {
      throw new InvalidNamespaceError(
        `Invalid namespace label '${label}' found in ${namespace}. Namespace labels ` +
          `must be strings, but got ${typeof label}.`
      );
    }
    if (label.includes(".")) {
      throw new InvalidNamespaceError(
        `Invalid namespace label '${label}' found in ${namespace}. Namespace labels cannot contain periods ('.').`
      );
    }
    if (label === "") {
      throw new InvalidNamespaceError(
        `Namespace labels cannot be empty strings. Got ${label} in ${namespace}`
      );
    }
  }
  if (namespace[0] === "langgraph") {
    throw new InvalidNamespaceError(
      `Root label for namespace cannot be "langgraph". Got: ${namespace}`
    );
  }
}

/**
 * Configuration for MongoDB Atlas Vector Search index.
 */
export type IndexConfig = {
  /** Atlas vector search index name */
  name: string;
  /**
   * Embedding dimensionality. Required when using client-side embeddings so that
   * the store can validate vector dimensions. Not required (and ignored) in Atlas
   * auto-embedding mode, where the model determines dimensionality automatically.
   */
  dims?: number;
  /** Sub-field of value to embed (e.g. "content"). If omitted, the entire value object is serialized and embedded. */
  embeddingKey?: string;
  /** Similarity function for Atlas vector search (default: "cosine") */
  similarityFunction?: "cosine" | "euclidean" | "dotProduct";
  /** Filter fields to declare in the Atlas vector search index */
  filters?: string[];
  /**
   * Field path used in the Atlas $vectorSearch stage.
   * - For client-side embeddings: defaults to "embedding" (the field written by the store).
   * - For Atlas auto-embedding (no embeddings provided): set this to the text field
   *   to be embedded (e.g. "value.content").
   */
  path?: string;
  /**
   * Voyage AI model to use for auto-embedding (e.g. "voyage-3", "voyage-4").
   * Required when using Atlas auto-embedding (no embeddings provided).
   * Ignored when using client-side embeddings.
   */
  model?: string;
  /**
   * Modality of the content to embed. Defaults to "text".
   * Only used in Atlas auto-embedding mode.
   */
  modality?: string;
};

/**
 * Time-to-live configuration for automatic document expiration.
 */
export type TTLConfig = {
  /** Default TTL in seconds */
  defaultTtl: number;
  /** Whether to reset TTL timer on Get (extends life of frequently accessed items) */
  refreshOnRead: boolean;
};

export type MongoDBStoreParams = {
  client: MongoClient;
  dbName?: string;
  collectionName?: string;
  enableTimestamps?: boolean;
  embeddings?: EmbeddingsInterface;
  indexConfig?: IndexConfig;
  ttl?: TTLConfig;
};

/**
 * A long-term persistent key-value store backed by MongoDB.
 * Supports hierarchical namespace organization with Put, Get, ListNamespaces, and Search operations.
 * Optional vector search via embeddings or MongoDB Atlas Vector Search.
 */
export class MongoDBStore extends BaseStore {
  protected client: MongoClient;
  protected db: MongoDatabase;
  collectionName = "store";
  protected enableTimestamps: boolean;
  protected embeddings?: EmbeddingsInterface;
  protected indexConfig?: IndexConfig;
  protected ttl?: TTLConfig;

  private get timestampOp() {
    return this.enableTimestamps
      ? ({ $currentDate: { upserted_at: true } } as const)
      : {};
  }

  constructor({
    client,
    dbName,
    collectionName,
    enableTimestamps,
    embeddings,
    indexConfig,
    ttl,
  }: MongoDBStoreParams) {
    super();
    this.client = client;
    this.client.appendMetadata({
      name: "langgraphjs_store",
    });
    this.db = this.client.db(dbName);
    this.collectionName = collectionName ?? this.collectionName;
    this.enableTimestamps = enableTimestamps ?? false;
    this.embeddings = embeddings;
    this.indexConfig = indexConfig;
    this.ttl = ttl;
  }

  /**
   * Factory method to create a MongoDBStore from a connection string.
   * Automatically creates and connects a MongoDB client.
   * @param connString MongoDB connection string
   * @param params Store configuration
   */
  static async fromConnString(
    connString: string,
    params?: Omit<MongoDBStoreParams, "client">
  ): Promise<MongoDBStore> {
    const { MongoClient: MC } = await import("mongodb");
    const client = new MC(connString);
    await client.connect();
    const store = new MongoDBStore({ ...params, client });
    await store.start();
    return store;
  }

  /**
   * Execute a batch of operations (Put, Get, ListNamespaces, Search).
   * Operations are processed in order, but multiple consecutive Puts are batched.
   * PutOperations are deduplicated by (namespace, key) — last write wins.
   * Embeddings are batched in a single embedDocuments call.
   */
  async batch<Op extends readonly Operation[]>(
    operations: Op
  ): Promise<OperationResults<Op>> {
    const results: unknown[] = new Array(operations.length);

    // Process operations in order, batching consecutive Puts
    let i = 0;
    while (i < operations.length) {
      const op = operations[i];

      // If it's a Put, collect all consecutive Puts
      if ("value" in op) {
        const putIndices: number[] = [i];
        let j = i + 1;
        while (j < operations.length && "value" in operations[j]) {
          putIndices.push(j);
          j++;
        }

        // Execute batched Puts
        await this.batchPuts(
          putIndices.map((idx) => ({
            index: idx,
            op: operations[idx] as PutOperation,
          })),
          results
        );

        i = j;
      } else {
        // Process single read operation
        const readOp = op;
        if ("key" in readOp && !("namespacePrefix" in readOp)) {
          results[i] = await this.getOp(readOp as GetOperation);
        } else if ("namespacePrefix" in readOp) {
          results[i] = await this.searchOp(readOp as SearchOperation);
        } else {
          results[i] = await this.listNamespacesOp(
            readOp as ListNamespacesOperation
          );
        }
        i++;
      }
    }

    return results as OperationResults<Op>;
  }

  /**
   * Execute a batch of Put operations with deduplication and batched embeddings.
   */
  private async batchPuts(
    putOpsWithIndex: Array<{ index: number; op: PutOperation }>,
    results: unknown[]
  ): Promise<void> {
    // Deduplicate put operations by (namespace, key) — last write wins
    const deduped = new Map<string, { index: number; op: PutOperation }>();
    for (const { index, op } of putOpsWithIndex) {
      const key = JSON.stringify({ namespace: op.namespace, key: op.key });
      deduped.set(key, { index, op });
    }

    // Prepare bulk write operations
    if (deduped.size > 0) {
      const bulkOps: any[] = [];
      const opsList = Array.from(deduped.values());

      // If embeddings are configured, batch all embedDocuments calls
      // Map embedding index to operation index to handle per-item index overrides
      const embeddingMap = new Map<number, number>();
      const textsToEmbed: string[] = [];

      // When embeddings are NOT provided, we assume MongoDB Atlas auto-embedding is enabled.
      // Documents are stored without an embedding field, and MongoDB will generate embeddings
      // automatically. Semantic search is still supported via Atlas $vectorSearch with
      // query.text (see searchOp / atlasVectorSearch for details).
      if (this.embeddings) {
        for (let i = 0; i < opsList.length; i++) {
          const { op } = opsList[i];

          // Check per-item index override
          const shouldIndex = op.index !== false && op.value !== null;
          if (!shouldIndex) {
            continue;
          }

          // Determine which fields to embed
          let textContent: string;
          if (Array.isArray(op.index)) {
            // Embed only specified fields
            const fields: Record<string, any> = {};
            for (const field of op.index) {
              fields[field] = op.value?.[field];
            }
            textContent = JSON.stringify(fields);
          } else {
            // Embed entire value
            textContent = JSON.stringify(op.value);
          }

          embeddingMap.set(textsToEmbed.length, i);
          textsToEmbed.push(textContent);
        }
      }

      let embeddingsList: number[][] = [];
      if (textsToEmbed.length > 0 && this.embeddings) {
        embeddingsList = await this.embeddings.embedDocuments(textsToEmbed);
      }

      // Build bulk write operations
      for (let i = 0; i < opsList.length; i++) {
        const { op } = opsList[i];
        const { namespace, key, value } = op;

        // Validate namespace
        validateNamespace(namespace);

        // Handle delete
        if (value === null) {
          bulkOps.push({
            deleteOne: {
              filter: { namespace, key },
            },
          });
          continue;
        }

        // Build document
        const now = new Date();
        const doc: Record<string, any> = {
          namespace,
          key,
          value, // Store JSON directly
          namespacePath: computeNamespacePath(namespace),
          updatedAt: now,
        };

        // Add embedding if available and this operation was embedded
        const embeddingIndex = Array.from(embeddingMap.entries()).find(
          ([_, opIdx]) => opIdx === i
        )?.[0];
        if (
          embeddingIndex !== undefined &&
          embeddingIndex < embeddingsList.length
        ) {
          doc.embedding = embeddingsList[embeddingIndex];
        }

        // Set expiration time if TTL is configured
        if (this.ttl) {
          doc.expiresAt = new Date(now.getTime() + this.ttl.defaultTtl * 1000);
        }

        bulkOps.push({
          updateOne: {
            filter: { namespace, key },
            update: {
              $set: doc,
              $setOnInsert: { createdAt: now },
              ...this.timestampOp,
            },
            upsert: true,
          },
        });
      }

      // Execute bulk write
      if (bulkOps.length > 0) {
        await this.db.collection(this.collectionName).bulkWrite(bulkOps);
      }

      // Fill results for put operations (all are undefined)
      for (const { index } of putOpsWithIndex) {
        results[index] = undefined;
      }
    }
  }

  /**
   * Retrieve an item by namespace and key.
   * If refreshOnRead is enabled, resets the TTL timer.
   */
  private async getOp(op: GetOperation): Promise<Item | null> {
    const { namespace, key } = op;

    let doc;
    if (this.ttl?.refreshOnRead) {
      // Use findOneAndUpdate to refresh the TTL timer
      const now = new Date();
      const updateDoc: Record<string, any> = { updatedAt: now };
      if (this.ttl) {
        updateDoc.expiresAt = new Date(
          now.getTime() + this.ttl.defaultTtl * 1000
        );
      }
      doc = await this.db
        .collection(this.collectionName)
        .findOneAndUpdate(
          { namespace, key },
          { $set: updateDoc },
          { returnDocument: "after" }
        );
    } else {
      doc = await this.db
        .collection(this.collectionName)
        .findOne({ namespace, key });
    }

    if (!doc) {
      return null;
    }

    return {
      value: doc.value,
      key: doc.key,
      namespace: doc.namespace,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
    };
  }

  /**
   * List namespaces in the store, optionally filtered by prefix/suffix/depth.
   */
  private async listNamespacesOp(
    op: ListNamespacesOperation
  ): Promise<string[][]> {
    const { limit, offset, namespacePrefix, maxDepth } =
      op as ListNamespacesOperation & {
        namespacePrefix?: string[];
        maxDepth?: number;
      };

    // Build aggregation pipeline
    const pipeline: Record<string, any>[] = [];

    // Stage 1: Match by namespace prefix if provided
    if (namespacePrefix && namespacePrefix.length > 0) {
      pipeline.push({
        $match: {
          $expr: {
            $eq: [
              { $slice: ["$namespace", namespacePrefix.length] },
              namespacePrefix,
            ],
          },
        },
      });
    }

    // Stage 2: Group to get unique namespaces
    pipeline.push({
      $group: { _id: "$namespace" },
    });

    // Stage 3: Filter by max depth if provided
    if (maxDepth !== undefined) {
      pipeline.push({
        $match: {
          $expr: {
            $lte: [{ $size: "$_id" }, maxDepth],
          },
        },
      });
    }

    // Stage 4: Sort by namespace
    pipeline.push({
      $sort: { _id: 1 },
    });

    // Stage 5: Apply offset and limit
    if (offset !== undefined && offset > 0) {
      pipeline.push({
        $skip: offset,
      });
    }

    pipeline.push({
      $limit: limit ?? 100,
    });

    // Execute aggregation
    const docs = await this.db
      .collection(this.collectionName)
      .aggregate(pipeline)
      .toArray();

    // Extract namespace arrays from aggregation results
    const result = docs.map((doc) => doc._id);

    return result;
  }

  /**
   * Search for items by namespace prefix and filter criteria.
   * Supports both field-based filtering (structured search) and vector similarity search.
   *
   * Field-based search (filter parameter): Always available, searches value fields.
   * Vector search (query parameter): Requires embeddings to be configured.
   *   - If embeddings are provided: Performs semantic search on the query
   *   - If embeddings are NOT provided: Throws an error. In this mode, MongoDB Atlas
   *     auto-embedding is assumed to be enabled, but semantic search is not supported.
   *     Use field-based filtering instead.
   */
  private async searchOp(op: SearchOperation): Promise<SearchItem[]> {
    const {
      namespacePrefix,
      filter,
      limit = 100,
      offset = 0,
      query,
    } = op as SearchOperation & { query?: string };

    // If query is provided, vector search is requested
    if (query) {
      // Vector search requires either client-side embeddings or an Atlas indexConfig
      // (which enables Atlas auto-embedding via $vectorSearch query.text).
      if (!this.embeddings && !this.indexConfig) {
        throw new Error(
          "Vector search requires either an embeddings interface or an indexConfig. " +
            "Provide an embeddings interface for client-side embedding, or provide an " +
            "indexConfig to use MongoDB Atlas auto-embedding (query.text mode)."
        );
      }
      return this.vectorSearch(query, namespacePrefix || [], limit, offset);
    }

    // Otherwise, perform structured field-based search
    // Build MongoDB query for namespace and field filtering
    const mongoQuery: Record<string, any> = {};

    // Filter by namespace if provided (exact match)
    if (namespacePrefix && namespacePrefix.length > 0) {
      mongoQuery.namespace = namespacePrefix;
    }

    // Build filter conditions against the stored value document
    if (filter && Object.keys(filter).length > 0) {
      for (const [field, condition] of Object.entries(filter)) {
        if (
          typeof condition === "object" &&
          condition !== null &&
          !Array.isArray(condition)
        ) {
          // Handle advanced operators
          for (const [operator, operatorValue] of Object.entries(condition)) {
            switch (operator) {
              case "$eq":
                mongoQuery[`value.${field}`] = operatorValue;
                break;
              case "$ne":
                mongoQuery[`value.${field}`] = { $ne: operatorValue };
                break;
              case "$gt":
                mongoQuery[`value.${field}`] = { $gt: operatorValue };
                break;
              case "$gte":
                mongoQuery[`value.${field}`] = { $gte: operatorValue };
                break;
              case "$lt":
                mongoQuery[`value.${field}`] = { $lt: operatorValue };
                break;
              case "$lte":
                mongoQuery[`value.${field}`] = { $lte: operatorValue };
                break;
            }
          }
        } else {
          // Exact match
          mongoQuery[`value.${field}`] = condition;
        }
      }
    }

    // Execute query with limit and offset
    const docs = await this.db
      .collection(this.collectionName)
      .find(mongoQuery)
      .skip(offset)
      .limit(limit)
      .toArray();

    // Deserialize only the matched documents
    const items: SearchItem[] = [];
    for (const doc of docs) {
      items.push({
        value: doc.value,
        key: doc.key,
        namespace: doc.namespace,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
      });
    }

    return items;
  }

  /**
   * Perform vector similarity search.
   * Uses Atlas $vectorSearch if indexConfig is set:
   *   - With client-side embeddings (embeddings configured): passes queryVector.
   *   - With Atlas auto-embedding (no embeddings, indexConfig set): passes query.text,
   *     and Atlas generates the query embedding server-side via Voyage AI.
   * Falls back to in-memory cosine similarity when only embeddings are configured.
   */
  private async vectorSearch(
    query: string,
    namespacePrefix: string[],
    limit: number,
    offset: number
  ): Promise<SearchItem[]> {
    // If indexConfig is set, use Atlas $vectorSearch
    if (this.indexConfig) {
      // Generate a client-side query embedding if embeddings are configured;
      // otherwise Atlas auto-embedding will handle it via query.text.
      const queryEmbedding = this.embeddings
        ? await this.embeddings.embedQuery(query)
        : null;
      return this.atlasVectorSearch(
        query,
        queryEmbedding,
        namespacePrefix,
        limit,
        offset
      );
    }

    // Otherwise, use in-memory cosine similarity (requires client-side embeddings)
    const queryEmbedding = await this.embeddings!.embedQuery(query);
    return this.inMemoryVectorSearch(
      queryEmbedding,
      namespacePrefix,
      limit,
      offset
    );
  }

  /**
   * Perform vector search using MongoDB Atlas $vectorSearch aggregation stage.
   *
   * When queryEmbedding is provided (client-side embedding mode), passes it as
   * queryVector. When queryEmbedding is null (Atlas auto-embedding mode), passes
   * queryText via query.text so Atlas generates the embedding server-side.
   */
  private async atlasVectorSearch(
    queryText: string,
    queryEmbedding: number[] | null,
    namespacePrefix: string[],
    limit: number,
    offset: number
  ): Promise<SearchItem[]> {
    const pipeline: Record<string, any>[] = [];

    // Build namespace prefix filter string for vector search
    let namespacePrefixFilter = "";
    if (namespacePrefix && namespacePrefix.length > 0) {
      namespacePrefixFilter = namespacePrefix.join("/");
    }

    // Resolve the field path for $vectorSearch:
    // - client-side embeddings: stored in the "embedding" field by default
    // - Atlas auto-embedding: the text field configured in the Atlas index definition
    const path = this.indexConfig!.path ?? "embedding";

    // Stage 1: Vector search with namespace filtering
    const vectorSearchStage: Record<string, any> = {
      $vectorSearch: {
        index: this.indexConfig!.name,
        path,
        numCandidates: Math.max(limit + offset, 100) * 10,
        limit: limit + offset,
      },
    };

    // Use queryVector for client-side embeddings, or query.text for Atlas auto-embedding
    if (queryEmbedding !== null) {
      vectorSearchStage.$vectorSearch.queryVector = queryEmbedding;
    } else {
      vectorSearchStage.$vectorSearch.query = { text: queryText };
    }

    // Add namespace filter if provided ($vectorSearch filter uses MQL query syntax)
    if (namespacePrefixFilter) {
      vectorSearchStage.$vectorSearch.filter = {
        namespacePath: { $in: [namespacePrefixFilter] },
      };
    }

    pipeline.push(vectorSearchStage);

    // Stage 2: Add similarity score metadata
    pipeline.push({
      $addFields: {
        score: { $meta: "vectorSearchScore" },
      },
    });

    // Stage 3: Skip for offset
    if (offset > 0) {
      pipeline.push({
        $skip: offset,
      });
    }

    // Stage 4: Limit results
    pipeline.push({
      $limit: limit,
    });

    // Execute aggregation
    const docs = await this.db
      .collection(this.collectionName)
      .aggregate(pipeline)
      .toArray();

    // Deserialize values from matched documents
    const items: SearchItem[] = [];
    for (const doc of docs) {
      const value = doc.value as Record<string, any>;

      items.push({
        value,
        key: doc.key,
        namespace: doc.namespace,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
        score: doc.score,
      });
    }

    return items;
  }

  /**
   * Perform vector search using in-memory cosine similarity.
   */
  private async inMemoryVectorSearch(
    queryEmbedding: number[],
    namespacePrefix: string[],
    limit: number,
    offset: number
  ): Promise<SearchItem[]> {
    // Build query for namespace match (exact)
    const mongoQuery: Record<string, any> = {};
    if (namespacePrefix && namespacePrefix.length > 0) {
      mongoQuery.namespace = namespacePrefix;
    }

    // Find similar vectors using simple distance calculation
    const vectorDocs = await this.db
      .collection(this.collectionName)
      .find(mongoQuery)
      .toArray();

    // Filter to documents that have embeddings
    const docsWithEmbeddings = vectorDocs.filter((doc) => doc.embedding);

    // Calculate similarity scores (cosine distance)
    const similarities = docsWithEmbeddings.map((doc: any) => {
      const similarity = this.cosineSimilarity(queryEmbedding, doc.embedding);
      return { similarity, doc };
    });

    // Sort by similarity and apply limit/offset
    const sorted = similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(offset, offset + limit);

    // Deserialize values from matched documents
    const items: SearchItem[] = [];
    for (const { doc, similarity } of sorted) {
      const value = doc.value as Record<string, any>;

      items.push({
        value,
        key: doc.key,
        namespace: doc.namespace,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
        score: similarity,
      });
    }

    return items;
  }

  /**
   * Calculate cosine similarity between two vectors.
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must have the same length");
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }

  /**
   * Initialize the store: creates the {namespace, key} unique index, an optional
   * TTL index, and — when indexConfig is provided — the vector search index.
   */
  async start(): Promise<void> {
    const collection = this.db.collection(this.collectionName);

    await collection.createIndex({ namespace: 1, key: 1 }, { unique: true });

    if (this.ttl) {
      await collection.createIndex(
        { expiresAt: 1 },
        { expireAfterSeconds: 0 }
      );
    }

    if (this.indexConfig) {
      const fields: Record<string, any>[] = [];

      if (this.embeddings) {
        // Client-side embedding mode: standard vector field on the embedding field
        const vectorField: Record<string, any> = {
          type: "vector",
          path: this.indexConfig.path ?? "embedding",
          similarity: this.indexConfig.similarityFunction ?? "cosine",
        };
        if (this.indexConfig.dims !== undefined) {
          vectorField.numDimensions = this.indexConfig.dims;
        }
        fields.push(vectorField);
      } else {
        // Atlas auto-embedding mode: let MongoDB embed the text field via Voyage AI
        fields.push({
          type: "autoEmbed",
          path: this.indexConfig.path,
          model: this.indexConfig.model,
          modality: this.indexConfig.modality ?? "text",
        });
      }

      // Always index namespacePath so $vectorSearch pre-filtering is efficient
      fields.push({ type: "filter", path: "namespacePath" });

      // Any additional filter fields declared by the caller
      for (const filterField of this.indexConfig.filters ?? []) {
        if (filterField !== "namespacePath") {
          fields.push({ type: "filter", path: filterField });
        }
      }

      try {
        await collection.createSearchIndex({
          name: this.indexConfig.name,
          type: "vectorSearch",
          definition: { fields },
        } as any);
      } catch (err: any) {
        // Ignore "index already exists" — treat createSearchIndex as idempotent
        if (!err?.message?.toLowerCase().includes("already exists")) {
          throw err;
        }
      }
    }
  }

  /**
   * Clean up the store (no-op for now).
   */
  async stop(): Promise<void> {
    // No-op
  }
}
