from src.ground_truth_extractor import GroundTruthExtractor

if __name__ == '__main__':
    run_id = "20250503_080859"
    biomes = 1
    trajectories = 50
    frames = 1
    gt_extractor = GroundTruthExtractor(run_id=run_id, biomes_count=biomes, trajectories_count=trajectories, frames_count=frames)
    gt = gt_extractor.extract_absolute_distances()
    gt = gt_extractor.extract_relative_distances()
    gt = gt_extractor.extract_relative_directions()