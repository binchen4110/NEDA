class QueryPastVocabulary:
    def __init__(self, first_seen, cutoff):
        self.first_seen = first_seen
        self.cutoff = cutoff
        self.cache = {}

    def get(self, key, default=()):
        if key not in self.cache:
            pair = self.first_seen.get(key)
            if pair is None:
                return default
            objects, first_times = pair
            self.cache[key] = tuple(sorted(objects[first_times < self.cutoff].tolist()))
        return self.cache[key] or default
