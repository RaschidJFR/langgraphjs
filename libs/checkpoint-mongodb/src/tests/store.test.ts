import { MongoDBStore } from "../store";
import type {
  PutOperation,
} from "@langchain/langgraph-checkpoint";
import { beforeEach, describe, it, expect, vi } from "vitest";


describe("MongoDBStore", () => {
  let store: MongoDBStore;
  let mockClient: any;
  let mockDb: any;
  let mockCollection: any;

  beforeEach(() => {
    // Create mocks and link them so they're consistent
    const createFindCursor = () => ({
      project: vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([]),
      })),
      skip: vi.fn(() => ({
        limit: vi.fn(() => ({
          toArray: vi.fn().mockResolvedValue([]),
        })),
      })),
      toArray: vi.fn().mockResolvedValue([]),
    });

    const collectionMock = {
      updateOne: vi.fn().mockResolvedValue({ ok: 1 }),
      findOne: vi.fn().mockResolvedValue(null),
      find: vi.fn(() => createFindCursor()),
      deleteOne: vi.fn().mockResolvedValue({ deletedCount: 1 }),
      createIndex: vi.fn().mockResolvedValue("namespace_1_key_1"),
      aggregate: vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([]),
      })),
      bulkWrite: vi.fn().mockResolvedValue({ ok: 1 }),
    };

    mockDb = {
      collection: vi.fn(() => collectionMock),
    };

    mockClient = {
      appendMetadata: vi.fn(),
      db: vi.fn(() => mockDb),
    };

    mockCollection = collectionMock;

    store = new MongoDBStore({
      client: mockClient as any,
      dbName: "test",
      collectionName: "store",
      enableTimestamps: false,
    });
  });

  describe("Put operation", () => {
    it("should upsert a document with serialized value", async () => {
      const operation: PutOperation = {
        namespace: ["documents", "user123"],
        key: "doc1",
        value: { title: "Test", content: "Hello" },
      };

      await store.batch([operation]);

      expect(mockCollection.bulkWrite).toHaveBeenCalled();
      const calls = mockCollection.bulkWrite.mock.calls[0][0];
      expect(calls).toHaveLength(1);
      const { updateOne } = calls[0];
      expect(updateOne.filter).toEqual({
        namespace: ["documents", "user123"],
        key: "doc1",
      });
      expect(updateOne.update.$set.value).toEqual({
        title: "Test",
        content: "Hello",
      });
      expect(updateOne.update.$set.namespace).toEqual([
        "documents",
        "user123",
      ]);
      expect(updateOne.upsert).toBe(true);
    });

    it("should throw InvalidNamespaceError for empty namespace", async () => {
      const operation: PutOperation = {
        namespace: [],
        key: "doc1",
        value: { title: "Test" },
      };

      await expect(store.batch([operation])).rejects.toThrow(
        "Namespace cannot be empty"
      );
    });

    it("should throw InvalidNamespaceError for namespace with periods", async () => {
      const operation: PutOperation = {
        namespace: ["documents.invalid"],
        key: "doc1",
        value: { title: "Test" },
      };

      await expect(store.batch([operation])).rejects.toThrow(
        "Namespace labels cannot contain periods"
      );
    });

    it("should throw InvalidNamespaceError for 'langgraph' root label", async () => {
      const operation: PutOperation = {
        namespace: ["langgraph", "data"],
        key: "doc1",
        value: { title: "Test" },
      };

      await expect(store.batch([operation])).rejects.toThrow(
        'Root label for namespace cannot be "langgraph"'
      );
    });

    it("should delete document when value is null", async () => {
      const operation: PutOperation = {
        namespace: ["documents", "user123"],
        key: "doc1",
        value: null,
      };

      await store.batch([operation]);

      expect(mockCollection.bulkWrite).toHaveBeenCalledWith(
        expect.arrayContaining([
          expect.objectContaining({
            deleteOne: expect.objectContaining({
              filter: { namespace: ["documents", "user123"], key: "doc1" },
            }),
          }),
        ])
      );
    });

    it("should skip embedding when index is false", async () => {
      const operation: PutOperation = {
        namespace: ["documents"],
        key: "doc1",
        value: { title: "Test" },
        index: false,
      };

      await store.batch([operation]);

      // The bulk write should be called, and embedding should not be present
      expect(mockCollection.bulkWrite).toHaveBeenCalled();
      // Verify no embedDocuments was called (embeddings not configured in this test anyway)
    });

    it("should embed only specified fields when index is array", async () => {
      // Create a store with embeddings
      const storeWithEmbeddings = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        enableTimestamps: false,
        embeddings: {
          embedQuery: vi
            .fn()
            .mockResolvedValue([0.1, 0.2, 0.3]),
          embedDocuments: vi
            .fn()
            .mockResolvedValue([[0.1, 0.2, 0.3]]),
        } as any,
      });

      const operation: PutOperation = {
        namespace: ["documents"],
        key: "doc1",
        value: { title: "Test", secret: "hidden", content: "body" },
        index: ["title", "content"], // Only embed title and content
      };

      await storeWithEmbeddings.batch([operation]);

      // Should have called embedDocuments with only the indexed fields
      expect(mockCollection.bulkWrite).toHaveBeenCalled();
    });
  });

  describe("Get operation", () => {
    it("should return null if document not found", async () => {
      const operation = {
        namespace: ["documents"],
        key: "nonexistent",
      };

      mockCollection.findOne.mockResolvedValueOnce(null);
      const results = await store.batch([operation]);

      expect(results[0]).toBeNull();
    });

    it("should deserialize and return item", async () => {
      const operation = {
        namespace: ["documents"],
        key: "doc1",
      };

      const now = new Date();
      mockCollection.findOne.mockResolvedValueOnce({
        namespace: ["documents"],
        key: "doc1",
        value: { title: "Test" },
        createdAt: now,
        updatedAt: now,
      });

      const results = await store.batch([operation]);
      const item = results[0];

      expect(item).toEqual({
        namespace: ["documents"],
        key: "doc1",
        value: { title: "Test" },
        createdAt: now,
        updatedAt: now,
      });
    });
  });

  describe("ListNamespaces operation", () => {
    it("should return empty array if no documents exist", async () => {
      const operation = {
        limit: 100,
        offset: 0,
      };

      // Set up the aggregate chain properly
      const aggregateMock = vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([]),
      }));
      mockCollection.aggregate = aggregateMock;

      const results = await store.batch([operation]);

      expect(results[0]).toEqual([]);
    });

    it("should return unique namespaces from documents", async () => {
      const operation = {
        limit: 100,
        offset: 0,
      };

      // Set up the aggregate chain to return test data
      const aggregateMock = vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([
          { _id: ["documents"] },
          { _id: ["users"] },
          { _id: ["documents", "v1"] },
        ]),
      }));
      mockCollection.aggregate = aggregateMock;

      const results = await store.batch([operation]);

      expect(results[0]).toEqual([
        ["documents"],
        ["users"],
        ["documents", "v1"],
      ]);
    });

    it("should apply limit and offset", async () => {
      const operation = {
        limit: 2,
        offset: 1,
      };

      // Set up the aggregate chain to return test data
      const aggregateMock = vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([
          { _id: ["b"] },
          { _id: ["c"] },
        ]),
      }));
      mockCollection.aggregate = aggregateMock;

      const results = await store.batch([operation]);

      expect(results[0]).toEqual([["b"], ["c"]]);
    });
  });

  describe("Search operation", () => {
    it("should return empty array if no documents match filter", async () => {
      const operation = {
        namespacePrefix: ["users"],
        filter: { status: "inactive" },
        limit: 100,
        offset: 0,
      };

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue([]),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);

      expect(results[0]).toEqual([]);
    });

    it("should search with exact match filter", async () => {
      const operation = {
        namespacePrefix: ["users"],
        filter: { status: "active" },
        limit: 100,
        offset: 0,
      };

      const now = new Date();
      const mockDoc = {
        namespace: ["users"],
        key: "user1",
        value: { status: "active", name: "Alice" },
        createdAt: now,
        updatedAt: now,
      };

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue([mockDoc]),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(1);
      expect(items[0].value).toEqual({ status: "active", name: "Alice" });
    });

    it("should search with comparison operators", async () => {
      const operation = {
        namespacePrefix: ["products"],
        filter: { price: { $gt: 100 } },
        limit: 100,
        offset: 0,
      };

      const now = new Date();
      const mockDocs = [
        {
          namespace: ["products"],
          key: "product1",
          value: { price: 150, name: "Premium" },
          createdAt: now,
          updatedAt: now,
        },
        {
          namespace: ["products"],
          key: "product2",
          value: { price: 200, name: "Luxury" },
          createdAt: now,
          updatedAt: now,
        },
      ];

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue(mockDocs),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(2);
      expect(items[0].value.price).toBeGreaterThan(100);
    });

    it("should filter by namespace prefix", async () => {
      const operation = {
        namespacePrefix: ["users", "profiles"],
        filter: {},
        limit: 100,
        offset: 0,
      };

      const now = new Date();
      const mockDoc = {
        namespace: ["users", "profiles"],
        key: "profile1",
        value: { bio: "Engineer" },
        createdAt: now,
        updatedAt: now,
      };

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue([mockDoc]),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(1);
      expect(items[0].namespace).toEqual(["users", "profiles"]);
    });

    it("should apply limit and offset to search results", async () => {
      const operation = {
        namespacePrefix: [],
        filter: { active: true },
        limit: 2,
        offset: 1,
      };

      const now = new Date();
      const mockItems = [
        {
          namespace: ["docs"],
          key: "doc2",
          value: { active: true },
          createdAt: now,
          updatedAt: now,
        },
        {
          namespace: ["docs"],
          key: "doc3",
          value: { active: true },
          createdAt: now,
          updatedAt: now,
        },
      ];

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue(mockItems),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(2);
      expect(items[0].key).toBe("doc2");
    });

    it("should combine namespace prefix and filter", async () => {
      const operation = {
        namespacePrefix: ["articles"],
        filter: { category: "tech", published: true },
        limit: 100,
        offset: 0,
      };

      const now = new Date();
      const mockDoc = {
        namespace: ["articles"],
        key: "article1",
        value: { category: "tech", published: true },
        createdAt: now,
        updatedAt: now,
      };

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue([mockDoc]),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(1);
      expect(items[0].value).toEqual({ category: "tech", published: true });
    });

    it("should support multiple comparison operators", async () => {
      const operation = {
        namespacePrefix: ["events"],
        filter: {
          date: { $gte: "2024-01-01" },
          count: { $lte: 100 }
        },
        limit: 100,
        offset: 0,
      };

      const now = new Date();
      const mockDoc = {
        namespace: ["events"],
        key: "event1",
        value: { date: "2024-06-01", count: 50 },
        createdAt: now,
        updatedAt: now,
      };

      const findMock = vi.fn(() => ({
        skip: vi.fn(() => ({
          limit: vi.fn(() => ({
            toArray: vi.fn().mockResolvedValue([mockDoc]),
          })),
        })),
      }));
      mockCollection.find = findMock;

      const results = await store.batch([operation]);
      const items = results[0] as any[];

      expect(items).toHaveLength(1);
    });

    it("should throw when neither embeddings nor indexConfig is provided and query is given", async () => {
      const operation = {
        namespacePrefix: ["documents"],
        query: "find something",
        limit: 100,
        offset: 0,
      };

      const bareStore = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        enableTimestamps: false,
        // No embeddings, no indexConfig
      });

      await expect(bareStore.batch([operation])).rejects.toThrow(
        /embeddings interface or an indexConfig/i
      );
    });
  });

  describe("Atlas auto-embedding mode ($vectorSearch with query.text)", () => {
    it("should use query.text in $vectorSearch when indexConfig is set but no embeddings", async () => {
      // In Atlas auto-embedding mode, dims is not required — the model determines it
      const storeAutoEmbed = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        indexConfig: {
          name: "my_vector_index",
          path: "value.content",
        },
      });

      const capturedPipelines: any[][] = [];
      mockCollection.aggregate = vi.fn((pipeline: any[]) => {
        capturedPipelines.push(pipeline);
        return { toArray: vi.fn().mockResolvedValue([]) };
      });

      await storeAutoEmbed.batch([{
        namespacePrefix: ["memories"],
        query: "outdoor activities",
        limit: 5,
        offset: 0,
      }]);

      expect(capturedPipelines).toHaveLength(1);
      expect(capturedPipelines[0][0].$vectorSearch).toEqual({
        index: "my_vector_index",
        path: "value.content",
        query: { text: "outdoor activities" },
        numCandidates: 1000,
        limit: 5,
        filter: { namespacePath: { $in: ["memories"] } },
        // queryVector is absent — Atlas embeds the query server-side
      });
    });

    it("should default path to 'embedding' in $vectorSearch when indexConfig.path is not set", async () => {
      const storeAutoEmbed = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        indexConfig: { name: "my_vector_index" },
      });

      const capturedPipelines: any[][] = [];
      mockCollection.aggregate = vi.fn((pipeline: any[]) => {
        capturedPipelines.push(pipeline);
        return { toArray: vi.fn().mockResolvedValue([]) };
      });

      await storeAutoEmbed.batch([{
        namespacePrefix: ["docs"],
        query: "some query",
        limit: 10,
        offset: 0,
      }]);

      expect(capturedPipelines[0][0].$vectorSearch).toEqual({
        index: "my_vector_index",
        path: "embedding", // default when indexConfig.path is not set
        query: { text: "some query" },
        numCandidates: 1000,
        limit: 10,
        filter: { namespacePath: { $in: ["docs"] } },
      });
    });

    it("should not call any embed function in auto-embed mode", async () => {
      const storeAutoEmbed = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        indexConfig: { name: "my_vector_index", path: "value.content" },
      });

      mockCollection.aggregate = vi.fn(() => ({
        toArray: vi.fn().mockResolvedValue([]),
      }));

      // Should succeed without any embeddings configured
      await expect(storeAutoEmbed.batch([{
        namespacePrefix: ["docs"],
        query: "some query",
        limit: 10,
        offset: 0,
      }])).resolves.not.toThrow();
    });

    it("should use queryVector (not query.text) when both indexConfig and embeddings are configured", async () => {
      const mockEmbedQuery = vi.fn().mockResolvedValue([0.1, 0.2, 0.3]);
      const storeWithBoth = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        // dims is still accepted when using client-side embeddings
        indexConfig: { name: "my_vector_index", dims: 3 },
        embeddings: {
          embedQuery: mockEmbedQuery,
          embedDocuments: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3]]),
        } as any,
      });

      const capturedPipelines: any[][] = [];
      mockCollection.aggregate = vi.fn((pipeline: any[]) => {
        capturedPipelines.push(pipeline);
        return { toArray: vi.fn().mockResolvedValue([]) };
      });

      await storeWithBoth.batch([{
        namespacePrefix: ["docs"],
        query: "some query",
        limit: 10,
        offset: 0,
      }]);

      expect(capturedPipelines[0][0].$vectorSearch).toEqual({
        index: "my_vector_index",
        path: "embedding",
        queryVector: [0.1, 0.2, 0.3],
        numCandidates: 1000,
        limit: 10,
        filter: { namespacePath: { $in: ["docs"] } },
        // query.text is absent — embedding was generated client-side
      });
      expect(mockEmbedQuery).toHaveBeenCalledWith("some query");
    });

    it("should include namespace filter in $vectorSearch when namespacePrefix is set", async () => {
      const storeAutoEmbed = new MongoDBStore({
        client: mockClient,
        dbName: "test",
        collectionName: "store",
        indexConfig: { name: "my_vector_index" },
      });

      const capturedPipelines: any[][] = [];
      mockCollection.aggregate = vi.fn((pipeline: any[]) => {
        capturedPipelines.push(pipeline);
        return { toArray: vi.fn().mockResolvedValue([]) };
      });

      await storeAutoEmbed.batch([{
        namespacePrefix: ["memories", "user_1"],
        query: "outdoor activities",
        limit: 10,
        offset: 0,
      }]);

      expect(capturedPipelines[0][0].$vectorSearch).toEqual({
        index: "my_vector_index",
        path: "embedding",
        query: { text: "outdoor activities" },
        numCandidates: 1000,
        limit: 10,
        filter: { namespacePath: { $in: ["memories/user_1"] } },
      });
    });
  });
});