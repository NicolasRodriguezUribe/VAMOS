# algorithm/config.py

class NSGAIIConfig:
    """
    Declarative configuration holder for NSGA-II.
    No logic here, just storing parameters.
    """
    def __init__(self):
        self._cfg = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs):
        # e.g. method="sbx", kwargs={"prob": 0.9, "eta": 20}
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        # e.g. method="pm", kwargs={"prob": "1/n", "eta": 20}
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs):
        # e.g. method="tournament", kwargs={"pressure": 2}
        self._cfg["selection"] = (method, kwargs)
        return self

    def survival(self, method: str):
        # e.g. method="nsga2"
        self._cfg["survival"] = method
        return self

    def engine(self, value: str):
        # e.g. "numpy", "numba", "moocore"
        self._cfg["engine"] = value
        return self

    def fixed(self) -> dict:
        """
        Return a snapshot (shallow copy) of the configuration.
        We intentionally do not resolve "1/n" here; the algorithm does it
        once it knows the problem's n_var.
        """
        return dict(self._cfg)


class MOEADConfig:
    """
    Declarative configuration holder for MOEA/D settings.
    Mirrors NSGA-II builder style so both algorithms can share patterns.
    """

    def __init__(self):
        self._cfg = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def neighbor_size(self, value: int):
        self._cfg["neighbor_size"] = value
        return self

    def delta(self, value: float):
        self._cfg["delta"] = value
        return self

    def replace_limit(self, value: int):
        self._cfg["replace_limit"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def aggregation(self, method: str, **kwargs):
        self._cfg["aggregation"] = (method, kwargs)
        return self

    def weight_vectors(self, *, path: str | None = None, divisions: int | None = None):
        """
        path: optional CSV file with precomputed weights. If it does not exist,
        it will be created automatically.
        divisions: optional simplex-lattice divisions parameter; when omitted the
        generator chooses the smallest value that produces >= pop_size vectors.
        """
        self._cfg["weight_vectors"] = {"path": path, "divisions": divisions}
        return self

    def engine(self, value: str):
        self._cfg["engine"] = value
        return self

    def fixed(self) -> dict:
        return dict(self._cfg)


class SMSEMOAConfig:
    """
    Declarative configuration holder for SMS-EMOA settings.
    """

    def __init__(self):
        self._cfg = {}

    def pop_size(self, value: int):
        self._cfg["pop_size"] = value
        return self

    def crossover(self, method: str, **kwargs):
        self._cfg["crossover"] = (method, kwargs)
        return self

    def mutation(self, method: str, **kwargs):
        self._cfg["mutation"] = (method, kwargs)
        return self

    def selection(self, method: str, **kwargs):
        self._cfg["selection"] = (method, kwargs)
        return self

    def reference_point(
        self,
        *,
        vector=None,
        offset: float = 0.1,
        adaptive: bool = True,
    ):
        """
        vector: explicit reference point (list/tuple). If omitted, it is computed
                as max(F) + offset per objective.
        offset: additive buffer applied to the data-driven reference point.
        adaptive: when True, the reference point expands whenever a solution exceeds it.
        """
        self._cfg["reference_point"] = {
            "vector": vector,
            "offset": offset,
            "adaptive": adaptive,
        }
        return self

    def engine(self, value: str):
        self._cfg["engine"] = value
        return self

    def fixed(self) -> dict:
        return dict(self._cfg)
