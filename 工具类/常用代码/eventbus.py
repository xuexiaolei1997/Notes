
class EventBus:
    """
    事件总线
    """
    def __init__(self) -> None:
        self.subscriptions = {}
    
    def subscribe(self, event_type, handler):
        """
        订阅
        """
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(handler)
    
    def unsubscribe(self, event_type):
        """
        取消订阅
        """
        if event_type not in self.subscriptions:
            raise Exception(f"事件：{event_type}未订阅")
        del self.subscriptions[event_type]
    
    def publish(self, event):
        """
        发布
        """
        event_type = type(event)
        if event_type in self.subscriptions:
            for handler in self.subscriptions[event_type]:
                handler(event)
