class CommunicationManager:
    def __init__(self):
        self.exchange_registry = {}
        

    def register_exchange(self, agent1_id, agent2_id, timestamp):
        key_pair = frozenset([agent1_id, agent2_id])
        self.exchange_registry[key_pair] = timestamp
        print(f"Agent {agent1_id} and Agent {agent2_id} exchanged context at timestamp {timestamp}")

    
    def has_exchanged(self, agent1_id, agent2_id):
        key_pair = frozenset([agent1_id, agent2_id])
        return key_pair in self.exchange_registry
    
    def get_last_exchange(self, agent1_id, agent2_id):
        key_pair = frozenset([agent1_id, agent2_id])
        return self.exchange_registry[key_pair]
    