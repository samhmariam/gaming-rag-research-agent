import unittest
import time
from typing import List, Dict, Any, Optional, Union, TypedDict

from lib.documents import Document, Corpus
from lib.memory import LongTermMemory, MemoryFragment, TimestampFilter, ShortTermMemory
from lib.state_machine import StateMachine, Step, EntryPoint, Termination


# ------------------------------
# Fakes for Vector Store layer
# ------------------------------

class FakeVectorStore:
    """
    Minimal in-memory stand-in for lib.vector_db.VectorStore.
    Provides add/query/get with simple term-matching ranking to avoid external services.
    """

    def __init__(self):
        self._docs: List[Document] = []

    def add(self, item: Union[Document, Corpus, List[Document]]):
        if isinstance(item, Document):
            self._docs.append(item)
        elif isinstance(item, list):
            for d in item:
                assert isinstance(d, Document)
                self._docs.append(d)
        elif isinstance(item, Corpus):
            for d in item:
                self._docs.append(d)
        else:
            raise TypeError("Unsupported type for add()")

    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict[str, Any]] = None, limit: Optional[int] = None):
        # Provide shape similar to Chroma get()
        matched = []
        for d in self._docs:
            if where:
                ok = True
                for k, v in where.items():
                    if d.metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            matched.append(d)
        if limit is not None:
            matched = matched[:limit]
        return {
            "documents": [doc.content for doc in matched],
            "metadatas": [doc.metadata for doc in matched],
            "ids": [doc.id for doc in matched if hasattr(doc, "id")],
        }

    def query(self, query_texts: List[str], n_results: int = 3, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None):
        q = (query_texts[0] if isinstance(query_texts, list) else str(query_texts)).lower()
        terms = [t for t in q.replace("?", " ").split() if t]

        def passes_where(meta: Dict[str, Any]) -> bool:
            if not where:
                return True
            # Support a subset of the "$and" with equality and basic timestamp comparisons used in LongTermMemory
            clauses = where.get("$and", [])
            for c in clauses:
                for key, cond in c.items():
                    if not isinstance(cond, dict):
                        if meta.get(key) != cond:
                            return False
                    else:
                        if "$eq" in cond and meta.get(key) != cond["$eq"]:
                            return False
                        if "$gt" in cond and not (meta.get(key, 0) > cond["$gt"]):
                            return False
                        if "$lt" in cond and not (meta.get(key, 0) < cond["$lt"]):
                            return False
            return True

        scored = []
        for d in self._docs:
            if not passes_where(d.metadata):
                continue
            content = (d.content or "").lower()
            score = sum(1 for t in terms if t in content)
            # Lower distance means closer; emulate cosine distance roughly with 1/(1+score)
            distance = 1.0 / (1.0 + float(score))
            scored.append((distance, d))

        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "documents": [[d.content for _, d in top]],
            "metadatas": [[d.metadata for _, d in top]],
            "distances": [[dist for dist, _ in top]],
        }


class FakeVectorStoreManager:
    def __init__(self):
        self._stores: Dict[str, FakeVectorStore] = {}

    def create_store(self, store_name: str, force: bool = False) -> FakeVectorStore:
        if force and store_name in self._stores:
            del self._stores[store_name]
        store = self._stores.get(store_name) or FakeVectorStore()
        self._stores[store_name] = store
        return store

    def get_or_create_store(self, store_name: str) -> FakeVectorStore:
        return self._stores.get(store_name) or self.create_store(store_name)


# ------------------------------
# Tests: LongTermMemory
# ------------------------------

class TestLongTermMemory(unittest.TestCase):
    def test_register_and_search_by_owner_and_namespace(self):
        manager = FakeVectorStoreManager()
        ltm = LongTermMemory(manager)  # type: ignore[arg-type]

        # Two owners in two namespaces
        ltm.register(MemoryFragment(content="UserA likes RPG games", owner="userA", namespace="prefs"))
        ltm.register(MemoryFragment(content="UserB plays racing games", owner="userB", namespace="prefs"))
        ltm.register(MemoryFragment(content="UserA favorite series is Pokemon", owner="userA", namespace="facts"))

        res = ltm.search(query_text="RPG", owner="userA", limit=5, namespace="prefs")
        texts = [f.content for f in res.fragments]
        self.assertTrue(any("RPG" in t for t in texts))
        self.assertTrue(all(f.owner == "userA" for f in res.fragments))
        self.assertTrue(all(f.namespace == "prefs" for f in res.fragments))

    def test_timestamp_filter(self):
        manager = FakeVectorStoreManager()
        ltm = LongTermMemory(manager)  # type: ignore[arg-type]

        old_ts = int(time.time()) - 60 * 60 * 24  # 24 hours ago
        new_ts = int(time.time())

        ltm.register(MemoryFragment(content="Old fact about release date", owner="userA", namespace="facts", timestamp=old_ts))
        ltm.register(MemoryFragment(content="New update for release date", owner="userA", namespace="facts", timestamp=new_ts))

        # Filter for memories within the last hour
        tsf = TimestampFilter(greater_than_value=int(time.time()) - 3600)
        res = ltm.search(query_text="release", owner="userA", limit=5, timestamp_filter=tsf, namespace="facts")
        texts = [f.content for f in res.fragments]
        self.assertIn("New update for release date", texts)
        self.assertNotIn("Old fact about release date", texts)


# ------------------------------
# Tests: ShortTermMemory
# ------------------------------

class TestShortTermMemory(unittest.TestCase):
    def test_sessions_and_last_object(self):
        stm = ShortTermMemory()
        stm.create_session("s1")
        self.assertIn("s1", stm.get_all_sessions())
        self.assertIsNone(stm.get_last_object("s1"))

        stm.add({"run": 1}, session_id="s1")
        stm.add({"run": 2}, session_id="s1")
        last = stm.get_last_object("s1")
        self.assertEqual(last["run"], 2)

        stm.reset("s1")
        self.assertIsNone(stm.get_last_object("s1"))


# ------------------------------
# Tests: StateMachine
# ------------------------------

class SMState(TypedDict, total=False):
    x: int
    path: List[str]


class TestStateMachine(unittest.TestCase):
    def test_linear_and_conditional_transitions(self):
        sm = StateMachine[SMState](SMState)
        entry = EntryPoint[SMState]()
        term = Termination[SMState]()

        def add_path(state: SMState, label: str) -> SMState:
            p = list(state.get("path", []))
            p.append(label)
            return {"path": p, "x": state.get("x", 0)}

        s1 = Step[SMState]("s1", lambda st: add_path(st, "s1"))
        s2 = Step[SMState]("s2", lambda st: add_path(st, "s2"))
        s3 = Step[SMState]("s3", lambda st: add_path(st, "s3"))

        sm.add_steps([entry, s1, s2, s3, term])
        sm.connect(entry, s1)
        sm.connect(s1, [s2, s3], lambda st: s2 if st.get("x", 0) > 0 else s3)
        sm.connect(s2, term)
        sm.connect(s3, term)

        run = sm.run({"x": 1, "path": []})
        final = run.get_final_state()
        self.assertIsNotNone(final)
        self.assertEqual(final["path"], ["s1", "s2"])  # chose s2 path

        run2 = sm.run({"x": 0, "path": []})
        final2 = run2.get_final_state()
        self.assertEqual(final2["path"], ["s1", "s3"])  # chose s3 path


if __name__ == "__main__":
    unittest.main(verbosity=2)
