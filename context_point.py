class contextPoint:
    def __init__(self, response, context, frame):
        self.response = response
        self.context = context
        self.frame = frame
    def get_context(self):
        return f"\n context for image: {self.context}, description and decision: {self.response.get('context', '')}"
    def get_frame(self):
        return self.frame