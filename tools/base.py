class Tool:
    name: str
    description: str

    def run(self, query: str) -> str:
        raise NotImplementedError
