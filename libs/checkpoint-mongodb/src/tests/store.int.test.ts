import { MongoClient } from "mongodb";
import { MongoDBStore } from "../store";
import {
  type PutOperation,
  type GetOperation,
  type ListNamespacesOperation,
  type SearchOperation,
} from "@langchain/langgraph-checkpoint";
import { getEnvironmentVariable } from "@langchain/core/utils/env";
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";

describe("MongoDBStore Integration Tests", () => {
  let client: MongoClient;
  let store: MongoDBStore;
  let storeWithEmbeddings: MongoDBStore;
  const dbName = "langgraph_test";
  const mongoUrl = getEnvironmentVariable("MONGODB_URL") || "mongodb://localhost:27017";

  // Simple embeddings for testing - creates consistent embeddings from text hash
  const createTestEmbeddings = (): EmbeddingsInterface => {
    const hashText = (text: string): number => {
      let hash = 0;
      for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash = hash & hash; // Convert to 32bit integer
      }
      return hash;
    };

    const textToEmbedding = (text: string): number[] => {
      const hash = hashText(text);
      const embedding = new Array(10).fill(0);
      for (let i = 0; i < 10; i++) {
        embedding[i] = Math.sin((hash + i) * 0.1);
      }
      return embedding;
    };

    return {
      embedDocuments: async (texts: string[]) => {
        return Promise.resolve(texts.map(textToEmbedding));
      },
      embedQuery: async (text: string) => {
        return Promise.resolve(textToEmbedding(text));
      },
    };
  };

  beforeAll(async () => {
    client = new MongoClient(mongoUrl, {
      auth: { username: "user", password: "password" },
    });
    await client.connect();

    // Create store with real MongoDB connection
    store = new MongoDBStore({
      client,
      dbName,
      collectionName: "test_store",
    });

    // Initialize (create indexes)
    await store.start();

    // Clean up any previous test data
    await client.db(dbName).collection("test_store").deleteMany({});
    await client.db(dbName).collection("test_store_vectors").deleteMany({});

    // Create store with embeddings for vector search tests
    storeWithEmbeddings = new MongoDBStore({
      client,
      dbName,
      collectionName: "test_store_embeddings",
      embeddings: createTestEmbeddings(),
    });

    await storeWithEmbeddings.start();
    await client.db(dbName).collection("test_store_embeddings").deleteMany({});
    await client.db(dbName).collection("test_store_embeddings_vectors").deleteMany({});
  });

  afterAll(async () => {
    // Clean up test data
    await client.db(dbName).collection("test_store").deleteMany({});
    await client.db(dbName).collection("test_store_embeddings").deleteMany({});
    await client.db(dbName).collection("test_store_embeddings_vectors").deleteMany({});
    await client.close();
  });

  describe("Put and Get workflow", () => {
    it("should store and retrieve an item", async () => {
      const putOp: PutOperation = {
        namespace: ["users", "profiles"],
        key: "user123",
        value: { name: "Alice", email: "alice@example.com" },
      };

      // Put the item
      await store.batch([putOp]);

      // Get the item
      const getOp: GetOperation = {
        namespace: ["users", "profiles"],
        key: "user123",
      };

      const results = await store.batch([getOp]);
      const item = results[0];

      expect(item).toBeDefined();
      expect(item?.value).toEqual({ name: "Alice", email: "alice@example.com" });
      expect(item?.namespace).toEqual(["users", "profiles"]);
      expect(item?.key).toEqual("user123");
    });

    it("should update an existing item", async () => {
      const putOp1: PutOperation = {
        namespace: ["docs"],
        key: "doc1",
        value: { title: "Original", version: 1 },
      };

      await store.batch([putOp1]);

      const putOp2: PutOperation = {
        namespace: ["docs"],
        key: "doc1",
        value: { title: "Updated", version: 2 },
      };

      await store.batch([putOp2]);

      const getOp: GetOperation = {
        namespace: ["docs"],
        key: "doc1",
      };

      const results = await store.batch([getOp]);
      const item = results[0];

      expect(item?.value.version).toBe(2);
      expect(item?.value.title).toBe("Updated");
    });

    it("should return null for non-existent item", async () => {
      const getOp: GetOperation = {
        namespace: ["nonexistent"],
        key: "missing",
      };

      const results = await store.batch([getOp]);

      expect(results[0]).toBeNull();
    });
  });

  describe("Delete workflow", () => {
    it("should delete an item when value is null", async () => {
      // Put an item first
      const putOp: PutOperation = {
        namespace: ["temp"],
        key: "item1",
        value: { data: "will delete" },
      };

      await store.batch([putOp]);

      // Delete it
      const deleteOp: PutOperation = {
        namespace: ["temp"],
        key: "item1",
        value: null,
      };

      await store.batch([deleteOp]);

      // Verify it's gone
      const getOp: GetOperation = {
        namespace: ["temp"],
        key: "item1",
      };

      const results = await store.batch([getOp]);

      expect(results[0]).toBeNull();
    });
  });

  describe("ListNamespaces workflow", () => {
    beforeAll(async () => {
      // Create test data with various namespaces
      const ops: PutOperation[] = [
        { namespace: ["threads"], key: "t1", value: { id: 1 } },
        { namespace: ["threads"], key: "t2", value: { id: 2 } },
        { namespace: ["threads", "messages"], key: "m1", value: { text: "hi" } },
        { namespace: ["threads", "messages"], key: "m2", value: { text: "bye" } },
        { namespace: ["users"], key: "u1", value: { name: "Alice" } },
        { namespace: ["users", "profiles"], key: "p1", value: { bio: "..." } },
      ];

      await store.batch(ops);
    });

    it("should list all unique namespaces", async () => {
      const listOp: ListNamespacesOperation = {
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([listOp]);
      const namespaces = results[0] as string[][];

      expect(namespaces).toContainEqual(["threads"]);
      expect(namespaces).toContainEqual(["threads", "messages"]);
      expect(namespaces).toContainEqual(["users"]);
      expect(namespaces).toContainEqual(["users", "profiles"]);
    });

    it("should apply limit correctly", async () => {
      const listOp: ListNamespacesOperation = {
        limit: 2,
        offset: 0,
      };

      const results = await store.batch([listOp]);
      const namespaces = results[0] as string[][];

      expect(namespaces.length).toBeLessThanOrEqual(2);
    });

    it("should apply offset correctly", async () => {
      const listOp1: ListNamespacesOperation = {
        limit: 100,
        offset: 0,
      };

      const listOp2: ListNamespacesOperation = {
        limit: 100,
        offset: 2,
      };

      const results1 = await store.batch([listOp1]);
      const results2 = await store.batch([listOp2]);

      const namespaces1 = results1[0] as string[][];
      const namespaces2 = results2[0] as string[][];

      expect(namespaces2.length).toBeLessThan(namespaces1.length);
    });
  });

  describe("Batch operations", () => {
    it("should execute multiple operations in one batch", async () => {
      const operations = [
        {
          namespace: ["batch"],
          key: "item1",
          value: { num: 1 },
        } as PutOperation,
        {
          namespace: ["batch"],
          key: "item2",
          value: { num: 2 },
        } as PutOperation,
        {
          namespace: ["batch"],
          key: "item1",
        } as GetOperation,
      ];

      const results = await store.batch(operations);

      expect(results[0]).toBeUndefined(); // Put returns undefined
      expect(results[1]).toBeUndefined(); // Put returns undefined
      expect(results[2]?.value).toEqual({ num: 1 }); // Get returns item
    });
  });

  describe("Per-item index override", () => {
    it("should skip embedding when index is false", async () => {
      const ops: PutOperation[] = [
        {
          namespace: ["indexed"],
          key: "doc1",
          value: { title: "Searchable", content: "This should be embedded" },
        },
        {
          namespace: ["indexed"],
          key: "doc2",
          value: { title: "NotSearchable", secret: "This should not be embedded" },
          index: false,
        },
      ];

      await storeWithEmbeddings.batch(ops);

      // Get both items
      const results = await storeWithEmbeddings.batch([
        { namespace: ["indexed"], key: "doc1" } as GetOperation,
        { namespace: ["indexed"], key: "doc2" } as GetOperation,
      ]);

      expect(results[0]).toBeDefined();
      expect(results[1]).toBeDefined();
      expect(results[0]?.value).toEqual({ title: "Searchable", content: "This should be embedded" });
      expect(results[1]?.value).toEqual({ title: "NotSearchable", secret: "This should not be embedded" });
    });

    it("should embed only specified fields when index is array", async () => {
      const ops: PutOperation[] = [
        {
          namespace: ["selective"],
          key: "article1",
          value: {
            title: "Great Article",
            content: "Long article content...",
            internalNotes: "Editor notes - should not be searched",
          },
          index: ["title", "content"], // Only index title and content
        },
      ];

      await storeWithEmbeddings.batch(ops);

      // Retrieve and verify storage
      const results = await storeWithEmbeddings.batch([
        { namespace: ["selective"], key: "article1" } as GetOperation,
      ]);

      expect(results[0]).toBeDefined();
      expect(results[0]?.value).toEqual({
        title: "Great Article",
        content: "Long article content...",
        internalNotes: "Editor notes - should not be searched",
      });
    });

    it("should handle mixed index configurations in one batch", async () => {
      const ops: PutOperation[] = [
        {
          namespace: ["mixed"],
          key: "always",
          value: { searchable: "yes" },
        },
        {
          namespace: ["mixed"],
          key: "never",
          value: { searchable: "no" },
          index: false,
        },
        {
          namespace: ["mixed"],
          key: "selective",
          value: { field1: "search me", field2: "ignore me" },
          index: ["field1"],
        },
      ];

      await storeWithEmbeddings.batch(ops);

      // Verify all were stored correctly
      const results = await storeWithEmbeddings.batch([
        { namespace: ["mixed"], key: "always" } as GetOperation,
        { namespace: ["mixed"], key: "never" } as GetOperation,
        { namespace: ["mixed"], key: "selective" } as GetOperation,
      ]);

      expect(results[0]?.value).toEqual({ searchable: "yes" });
      expect(results[1]?.value).toEqual({ searchable: "no" });
      expect(results[2]?.value).toEqual({ field1: "search me", field2: "ignore me" });
    });
  });

  describe("Search operation", () => {
    beforeAll(async () => {
      // Prepare search test data
      const searchOps: PutOperation[] = [
        // Products with various prices
        { namespace: ["products"], key: "prod1", value: { name: "Budget Item", price: 29.99, category: "electronics" } },
        { namespace: ["products"], key: "prod2", value: { name: "Standard Item", price: 99.99, category: "electronics" } },
        { namespace: ["products"], key: "prod3", value: { name: "Premium Item", price: 299.99, category: "electronics" } },
        { namespace: ["products"], key: "prod4", value: { name: "Book", price: 19.99, category: "books" } },

        // Users with different statuses
        { namespace: ["users"], key: "user1", value: { username: "alice", status: "active", score: 95 } },
        { namespace: ["users"], key: "user2", value: { username: "bob", status: "inactive", score: 42 } },
        { namespace: ["users"], key: "user3", value: { username: "charlie", status: "active", score: 87 } },

        // Nested namespace documents
        { namespace: ["users", "profiles"], key: "profile1", value: { bio: "Engineer", city: "SF" } },
        { namespace: ["users", "profiles"], key: "profile2", value: { bio: "Designer", city: "NYC" } },
      ];

      await store.batch(searchOps);
    });

    it("should search with exact match filter", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: { category: "electronics" },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) => item.value.category === "electronics")).toBe(true);
    });

    it("should search with comparison operator $gt", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: { price: { $gt: 100 } },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) => item.value.price > 100)).toBe(true);
    });

    it("should search with comparison operator $lte", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: { price: { $lte: 50 } },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) => item.value.price <= 50)).toBe(true);
    });

    it("should search with multiple filter conditions", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: {
          category: "electronics",
          price: { $gte: 50 }
        },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) =>
        item.value.category === "electronics" && item.value.price >= 50
      )).toBe(true);
    });

    it("should search with namespace prefix filtering", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["users", "profiles"],
        filter: {},
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) =>
        JSON.stringify(item.namespace) === JSON.stringify(["users", "profiles"])
      )).toBe(true);
    });

    it("should apply limit to search results", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: {},
        limit: 2,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeLessThanOrEqual(2);
    });

    it("should apply offset to search results", async () => {
      const searchOp1: SearchOperation = {
        namespacePrefix: ["users"],
        filter: { status: "active" },
        limit: 100,
        offset: 0,
      };

      const searchOp2: SearchOperation = {
        namespacePrefix: ["users"],
        filter: { status: "active" },
        limit: 100,
        offset: 1,
      };

      const results1 = await store.batch([searchOp1]);
      const results2 = await store.batch([searchOp2]);

      const items1 = results1[0] as any[];
      const items2 = results2[0] as any[];

      expect(items2.length).toBeLessThanOrEqual(items1.length);
    });

    it("should handle empty filter (return all in namespace)", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["users"],
        filter: {},
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) =>
        JSON.stringify(item.namespace) === JSON.stringify(["users"])
      )).toBe(true);
    });

    it("should return empty array for non-matching filter", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["products"],
        filter: { category: "nonexistent" },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items).toEqual([]);
    });

    it("should search with complex multi-field conditions", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["users"],
        filter: {
          status: "active",
          score: { $gte: 85 }
        },
        limit: 100,
        offset: 0,
      };

      const results = await store.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      expect(items.every((item: any) =>
        item.value.status === "active" && item.value.score >= 85
      )).toBe(true);
    });
  });

  describe("Vector search operations", () => {
    beforeAll(async () => {
      // Prepare vector search test data
      const vectorOps: PutOperation[] = [
        { namespace: ["documents"], key: "doc1", value: { content: "machine learning algorithms", topic: "AI" } },
        { namespace: ["documents"], key: "doc2", value: { content: "deep neural networks", topic: "AI" } },
        { namespace: ["documents"], key: "doc3", value: { content: "database indexing strategies", topic: "databases" } },
        { namespace: ["documents"], key: "doc4", value: { content: "SQL query optimization", topic: "databases" } },
      ];

      await storeWithEmbeddings.batch(vectorOps);
    });

    it("should perform vector similarity search", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["documents"],
        query: "machine learning",
        limit: 100,
        offset: 0,
      } as any;

      const results = await storeWithEmbeddings.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      // At least one result should have a score (similarity)
      expect(items[0]).toHaveProperty("score");
    });

    it("should return results ranked by similarity", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["documents"],
        query: "neural networks",
        limit: 100,
        offset: 0,
      } as any;

      const results = await storeWithEmbeddings.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeGreaterThan(0);
      // Check that scores are in descending order
      for (let i = 0; i < items.length - 1; i++) {
        expect(items[i].score).toBeGreaterThanOrEqual(items[i + 1].score);
      }
    });

    it("should apply limit to vector search results", async () => {
      const searchOp: SearchOperation = {
        namespacePrefix: ["documents"],
        query: "data",
        limit: 2,
        offset: 0,
      } as any;

      const results = await storeWithEmbeddings.batch([searchOp]);
      const items = results[0] as any[];

      expect(items.length).toBeLessThanOrEqual(2);
    });

    it("should apply offset to vector search results", async () => {
      const searchOp1: SearchOperation = {
        namespacePrefix: ["documents"],
        query: "network systems",
        limit: 100,
        offset: 0,
      } as any;

      const searchOp2: SearchOperation = {
        namespacePrefix: ["documents"],
        query: "network systems",
        limit: 100,
        offset: 1,
      } as any;

      const results1 = await storeWithEmbeddings.batch([searchOp1]);
      const results2 = await storeWithEmbeddings.batch([searchOp2]);

      const items1 = results1[0] as any[];
      const items2 = results2[0] as any[];

      expect(items2.length).toBeLessThanOrEqual(items1.length);
    });
  });
});

// ---------------------------------------------------------------------------
// Atlas auto-embedding integration tests
//
// Enable by setting TEST_MONGODB_AUTOEMBEDDING=true in your environment.
// Requires Community Edition 8.2+. See docker-compose.yml for setup.
// ---------------------------------------------------------------------------

const AUTOEMBEDDING_URL =
  process.env.MONGODB_URL ??
  "mongodb://user:password@127.0.0.1:27017/?directConnection=true&authSource=admin";
const AUTOEMBEDDING_DB = "langgraph_test";
const AUTOEMBEDDING_COLLECTION = "test_autoembedding";
const AUTOEMBEDDING_INDEX = "test_autoembedding_index";

/**
 * Run a search op and retry until results come back.
 * Auto-embeddings are generated asynchronously, so the first search after
 * inserting documents may return empty until Atlas finishes embedding them.
 */
async function searchWithRetry(
  store: MongoDBStore,
  op: SearchOperation,
  timeoutMs = 90_000
): Promise<any[]> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const results = (await store.batch([op]))[0] as any[];
    if (results.length > 0) return results;
    await new Promise((resolve) => setTimeout(resolve, 3_000));
  }
  throw new Error(
    `Search returned no results within ${timeoutMs}ms — auto-embeddings may have failed`
  );
}

describe.skipIf(!process.env.TEST_MONGODB_AUTOEMBEDDING)(
  "Atlas auto-embedding (query.text mode)",
  () => {
    let store: MongoDBStore;
    let client: MongoClient;

    beforeAll(async () => {
      client = new MongoClient(AUTOEMBEDDING_URL);
      await client.connect();

      // fromConnString calls start() which creates the vector search index
      store = await MongoDBStore.fromConnString(AUTOEMBEDDING_URL, {
        dbName: AUTOEMBEDDING_DB,
        collectionName: AUTOEMBEDDING_COLLECTION,
        indexConfig: {
          name: AUTOEMBEDDING_INDEX,
          path: "value.content",
          model: "voyage-4",
        },
      });

      // Clean slate
      await client
        .db(AUTOEMBEDDING_DB)
        .collection(AUTOEMBEDDING_COLLECTION)
        .deleteMany({});

      // On a fresh container the preview image sometimes sends an invalid
      // 'service_tier' parameter to Voyage AI on its first call.
      const probePut = [{
        namespace: ["__warmup__"],
        key: "probe",
        value: { content: "warmup probe" },
      }] as PutOperation[];
      const probeDelete = [{
        namespace: ["__warmup__"],
        key: "probe",
        value: null,
      }] as PutOperation[];
      const probeSearch = {
        namespacePrefix: ["__warmup__"],
        query: "warmup",
        limit: 1,
      };

      await store.batch(probePut);
      try {
        await searchWithRetry(store, probeSearch, 5_000);
      } catch {
        // Rate-limit active — discard the stuck probe, sleep out the window,
        // then re-insert so it embeds on a fresh change-stream event.
        await store.batch(probeDelete);
        await new Promise((r) => setTimeout(r, 5_000));
        await store.batch(probePut);
        await searchWithRetry(store, probeSearch, 5_000);
      }

      // Clean up probe
      await store.batch(probeDelete);

      // Insert test memories — Atlas auto-embeds them via Voyage AI
      const puts: PutOperation[] = [
        {
          namespace: ["memories", "alice"],
          key: "mem_1",
          value: { content: "Alice loves hiking in the mountains and camping outdoors." },
        },
        {
          namespace: ["memories", "alice"],
          key: "mem_2",
          value: { content: "Alice is a software engineer who enjoys building developer tools." },
        },
        {
          namespace: ["memories", "alice"],
          key: "mem_3",
          value: { content: "Alice recently adopted a golden retriever named Biscuit." },
        },
        {
          namespace: ["memories", "bob"],
          key: "mem_4",
          value: { content: "Bob loves mountain climbing and rock climbing." },
        },
      ];
      await store.batch(puts);
    }, 120_000);

    afterAll(async () => {
      await client
        .db(AUTOEMBEDDING_DB)
        .collection(AUTOEMBEDDING_COLLECTION)
        .deleteMany({});
      await client.close();
    });

    it("should create a vectorSearch index with autoEmbed field on start()", async () => {
      const indexes = (await client
        .db(AUTOEMBEDDING_DB)
        .collection(AUTOEMBEDDING_COLLECTION)
        .listSearchIndexes(AUTOEMBEDDING_INDEX)
        .toArray()) as any[];

      expect(indexes).toHaveLength(1);
      const def = indexes[0].latestDefinition;
      const embedField = def.fields.find((f: any) => f.type === "autoEmbed");
      expect(embedField).toBeDefined();
      expect(embedField.path).toBe("value.content");
      expect(embedField.model).toBe("voyage-4");
    });

    it("should store documents without a client-side embedding field", async () => {
      const doc = await client
        .db(AUTOEMBEDDING_DB)
        .collection(AUTOEMBEDDING_COLLECTION)
        .findOne({ key: "mem_1", namespacePath: "memories/alice" });

      expect(doc).toBeDefined();
      expect(doc!.embedding).toBeUndefined();
      expect(doc!.value.content).toBe(
        "Alice loves hiking in the mountains and camping outdoors."
      );
    });

    it("should return scored results for a semantic query", async () => {
      const results = await searchWithRetry(store, {
        namespacePrefix: ["memories", "alice"],
        query: "outdoor activities",
        limit: 3,
        offset: 0,
      } as any);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0]).toHaveProperty("score");
      expect(typeof results[0].score).toBe("number");
    });

    it("should rank the hiking memory highest for 'outdoor activities'", async () => {
      const results = await searchWithRetry(store, {
        namespacePrefix: ["memories", "alice"],
        query: "outdoor activities",
        limit: 3,
        offset: 0,
      } as any);

      expect(results[0].value.content).toContain("hiking");
      for (let i = 0; i < results.length - 1; i++) {
        expect(results[i].score).toBeGreaterThanOrEqual(results[i + 1].score);
      }
    });

    it("should scope results to the searched namespace prefix", async () => {
      const results = await searchWithRetry(store, {
        namespacePrefix: ["memories", "alice"],
        query: "outdoor activities",
        limit: 10,
        offset: 0,
      } as any);

      expect(
        results.every((r: any) => r.namespace[1] === "alice")
      ).toBe(true);
    });
  }
);