from minedojo.sim import InventoryItem


run_config = {
    "step_count": 10,
    #"same_action_limit": 10,
    "error_limit": 5,
    "check_distance_interval": 10,
    "min_distance": 0.2
}

task_config = {
    "easy_1_seed_3": {
        "task_id": "harvest",
        "target_names": "milk_bucket",    # Items to harvest
        "target_quantities": 1,          # Quantities to harvest
        "specified_biome": "plains",           # Biome where task takes place
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        "initial_mobs": "cow",
        "initial_mob_spawn_range_low": (0, 0, 1),
        "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 3,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "easy_1_seed_5": {
        "task_id": "harvest",
        "target_names": "milk_bucket",    # Items to harvest
        "target_quantities": 1,          # Quantities to harvest
        "specified_biome": "plains",           # Biome where task takes place
        #"spawn_rate": 0.99,
        #"spawn_range_low": (0, 0, 1),
        #"spawn_range_high": (0, 0, 1),
        "initial_mobs": "cow",
        "initial_mob_spawn_range_low": (0, 0, 1),
        "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 5,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "easy_2_seed_3": {
        "task_id": "harvest",
        "target_names": "log",
        "target_quantities": 1,
        "specified_biome": "forest",
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        # "initial_mobs": "cow",
        # "initial_mob_spawn_range_low": (0, 0, 1),
        # "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 3,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="iron_axe", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "easy_2_seed_5": {
        "task_id": "harvest",
        "target_names": "log",
        "target_quantities": 1,
        "specified_biome": "forest",
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        # "initial_mobs": "cow",
        # "initial_mob_spawn_range_low": (0, 0, 1),
        # "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 5,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="iron_axe", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -7, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "mid_1_seed_3": {
        "task_id": "harvest",
        "target_names": "milk_bucket",    # Items to harvest
        "target_quantities": 1,          # Quantities to harvest
        "specified_biome": "plains",           # Biome where task takes place
        #"spawn_rate": 0.99,
        #"spawn_range_low": (0, 0, 1),
        #"spawn_range_high": (0, 0, 1),
        "initial_mobs": "cow",
        "initial_mob_spawn_range_low": (-2, 0, 2),
        "initial_mob_spawn_range_high": (2, 0, 2),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 3,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "mid_1_seed_5": {
        "task_id": "harvest",
        "target_names": "milk_bucket",    # Items to harvest
        "target_quantities": 1,          # Quantities to harvest
        "specified_biome": "plains",           # Biome where task takes place
        #"spawn_rate": 0.99,
        #"spawn_range_low": (0, 0, 1),
        #"spawn_range_high": (0, 0, 1),
        "initial_mobs": "cow",
        "initial_mob_spawn_range_low": (-2, 0, 0),
        "initial_mob_spawn_range_high": (2, 0, 0),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 5,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "hard_2_seed_3": {
        "task_id": "harvest",
        "target_names": "log",
        "target_quantities": 1,
        "specified_biome": "forest",
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        # "initial_mobs": "cow",
        # "initial_mob_spawn_range_low": (0, 0, 1),
        # "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 3,
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
    "hard_2_seed_5": {
        "task_id": "harvest",
        "target_names": "log",
        "target_quantities": 1,
        "specified_biome": "forest",
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        # "initial_mobs": "cow",
        # "initial_mob_spawn_range_low": (0, 0, 1),
        # "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 5,
        "initial_weather": "clear",
        "start_position": {"x": -5, "y": 64, "z": -2, "pitch": 0, "yaw": 0} # x: negative values means right
    },
}