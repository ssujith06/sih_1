import simpy
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- Parameters ---
RANDOM_SEED = 42
SIM_DURATION = 500
INITIAL_DELAY_MINS = 30
FREIGHT_TRAIN_ID = "Freight-A"

# --- Track Network ---
TRACK_NETWORK = nx.DiGraph()
TRACK_NETWORK.add_edges_from([
    ('A', 'B', {'length': 50, 'speed_limit': 1.0, 'block': 'Block-1'}),
    ('B', 'C', {'length': 30, 'speed_limit': 1.0, 'block': 'Block-2'}),
    ('C', 'D', {'length': 20, 'speed_limit': 0.5, 'block': 'Crossing-1'}),
    ('D', 'G', {'length': 60, 'speed_limit': 1.0, 'block': 'Block-3'}),
    ('B', 'E', {'length': 40, 'speed_limit': 0.8, 'block': 'Bypass-1'}),
    ('E', 'F', {'length': 30, 'speed_limit': 0.8, 'block': 'Bypass-2'}),
    ('F', 'D', {'length': 40, 'speed_limit': 1.0, 'block': 'Block-4'}),
])

# Precompute travel times
for u, v, data in TRACK_NETWORK.edges(data=True):
    data['travel_time'] = data['length'] / data['speed_limit']

# --- Train Agent ---
class TrainAgent:
    def _init_(self, env, name, path, controller, track_resources, event_log):
        self.env = env
        self.name = name
        self.path = path
        self.controller = controller
        self.track_resources = track_resources
        self.event_log = event_log
        self.next_block_index = 0
        self.arrival_time = None
        self.env.process(self.run())

    def run(self):
        self.log_event("START")
        while self.next_block_index < len(self.path)-1:
            u, v = self.path[self.next_block_index], self.path[self.next_block_index+1]
            edge = TRACK_NETWORK[u][v]
            block = edge["block"]
            travel_time = edge["travel_time"]

            with self.track_resources[block].request() as req:
                yield req
                self.log_event(f"ENTER_BLOCK_{block}")

                # Delay only freight at Block-1
                if self.name == FREIGHT_TRAIN_ID and block == "Block-1" and self.controller.apply_delay:
                    self.log_event("DELAY_START")
                    yield self.env.timeout(INITIAL_DELAY_MINS)
                    self.log_event("DELAY_END")
                    yield self.env.process(self.controller.handle_delay(self))

                yield self.env.timeout(travel_time)
                self.log_event(f"EXIT_BLOCK_{block}")

            self.next_block_index += 1

        self.arrival_time = self.env.now
        self.log_event("ARRIVED")

    def log_event(self, event):
        self.event_log.append({"time": self.env.now, "train": self.name, "event": event})

# --- Traffic Controller ---
class TrafficController:
    def _init_(self, env, network, track_resources, event_log, apply_delay=False):
        self.env = env
        self.network = network
        self.track_resources = track_resources
        self.event_log = event_log
        self.apply_delay = apply_delay

    def handle_delay(self, delayed_train):
        """This is now a generator → works with env.process"""
        self.log_event("RECOMPUTE_SCHEDULE", delayed_train.name)
        yield self.env.timeout(5)  # recomputation takes 5 mins

        # reroute passenger trains
        for train in self.env.trains:
            if train != delayed_train and "Passenger" in train.name:
                try:
                    new_path = nx.shortest_path(
                        self.network,
                        source=train.path[train.next_block_index],
                        target="G",
                        weight="travel_time"
                    )
                    train.path = new_path
                    train.next_block_index = 0
                    self.log_event("REROUTED", train.name)
                except nx.NetworkXNoPath:
                    self.log_event("NO_ALT_PATH", train.name)

    def log_event(self, event, train):
        self.event_log.append({"time": self.env.now, "train": train, "event": event})

# --- Simulation Setup ---
def setup_and_run_simulation(apply_delay=False):
    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    track_resources = {edge[2]["block"]: simpy.Resource(env, capacity=1)
                       for edge in TRACK_NETWORK.edges(data=True)}
    event_log = []
    controller = TrafficController(env, TRACK_NETWORK, track_resources, event_log, apply_delay)

    trains_to_create = [
        {"name": "Passenger-1", "path": ["A","B","C","D","G"]},
        {"name": "Passenger-2", "path": ["A","B","C","D","G"]},
        {"name": FREIGHT_TRAIN_ID, "path": ["A","B","E","F","D","G"]}
    ]
    env.trains = [TrainAgent(env, t["name"], t["path"], controller, track_resources, event_log) for t in trains_to_create]

    env.run(until=SIM_DURATION)
    return pd.DataFrame(event_log)

# --- Visualization ---
def plot_gantt_side_by_side(baseline_df, delayed_df):
    colors = {"Passenger-1":"blue","Passenger-2":"green","Freight-A":"red"}
    fig, axes = plt.subplots(1,2,figsize=(20,7),sharey=True)

    for ax, df, title in zip(axes,[baseline_df,delayed_df],["Baseline (No Delay)","With Delay & Rerouting"]):
        for train in df['train'].unique():
            if train=="SYSTEM": continue
            events=df[df['train']==train].sort_values('time')
            for i in range(len(events)-1):
                if events.iloc[i]['event'].startswith("ENTER_BLOCK"):
                    block=events.iloc[i]['event'].replace("ENTER_BLOCK_","")
                    exits=events[(events['event']==f"EXIT_BLOCK_{block}") & (events['time']>events.iloc[i]['time'])]
                    if not exits.empty:
                        start,end=events.iloc[i]['time'],exits.iloc[0]['time']
                        ax.barh(block,end-start,left=start,color=colors.get(train,'gray'),edgecolor='black',alpha=0.7,label=train)
        handles,labels=ax.get_legend_handles_labels()
        unique=dict(zip(labels,handles))
        ax.legend(unique.values(),unique.keys())
        ax.set_title(title,fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel("Track Blocks")
        ax.grid(True,axis='x',linestyle='--',alpha=0.5)
    plt.suptitle("Train Occupancy Gantt Chart Comparison",fontsize=16,weight='bold')
    plt.tight_layout()
    plt.show()

# --- Run Simulations ---
if _name=="main_":
    baseline_log = setup_and_run_simulation(apply_delay=False)
    delayed_log = setup_and_run_simulation(apply_delay=True)

    # Compare arrival times
    baseline_arrivals = baseline_log[baseline_log['event']=="ARRIVED"].set_index("train")["time"]
    delayed_arrivals = delayed_log[delayed_log['event']=="ARRIVED"].set_index("train")["time"]
    comparison = pd.DataFrame({"Baseline Arrival":baseline_arrivals,"Delayed Arrival":delayed_arrivals})
    comparison["Delay Impact"] = comparison["Delayed Arrival"]-comparison["Baseline Arrival"]
    print(comparison)

    # Visualize
    plot_gantt_side_by_side(baseline_log,delayed_log)