class contextPoint:
    def __init__(self, response, context):
        self.response = response
        self.context = context
    def get_context(self):
        return f"\n context for image: {self.context}, description and decision: {self.response.get('context', '')}"