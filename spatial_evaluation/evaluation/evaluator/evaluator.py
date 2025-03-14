import numpy as np

class Evaluator:
    def __init__(self, task):
        self.task = task

    def evaluate_predictions(self, filtered_frames, predictions, ground_truths):
        if self.task == "absolute_distance" or self.task == "relative_distance":
            return self.distance_metric(filtered_frames, predictions, ground_truths)
        elif self.task == "relative_direction":
            return self.direction_metric(filtered_frames, predictions, ground_truths)

    def distance_metric(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the Mean Absolute Error (MAE) between two lists of distances
        """
        errors = []

        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            biome, trajectory, frame = str(biome), str(trajectory), str(frame)
            
            try:
                errors.append(np.abs(np.array(ground_truths[biome][trajectory][frame][0]) - np.array(predictions[biome][trajectory][frame])))
            # TODO: Handle the errors better
            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not same lenght")
                continue

        mae = np.mean(errors)

        return round(mae, 2)
    
    def direction_metric(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the accuracy of predicted directions compared to ground truth.
        """
        pass