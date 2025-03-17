import numpy as np

class Evaluator:
    def __init__(self, task):
        self.task = task

    def evaluate_predictions(self, filtered_frames, predictions, ground_truths):
        if self.task == "absolute_distance":
            return self.evaluate_absolute_distance(filtered_frames, predictions, ground_truths)
        elif self.task == "relative_distance":
            return self.evaluate_relative_distance(filtered_frames, predictions, ground_truths)
        elif self.task == "relative_direction":
            return self.evaluate_relative_direction(filtered_frames, predictions, ground_truths)

    def evaluate_absolute_distance(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the Mean Absolute Error (MAE) between predictions and ground truths
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
    
    def evaluate_relative_distance(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the Mean Absolute Error (MAE) between predictions and ground truths
        """
        errors = []

        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            biome, trajectory, frame = str(biome), str(trajectory), str(frame)
            
            try:
                errors.append(np.abs(np.array(ground_truths[biome][trajectory][frame]) - np.array(predictions[biome][trajectory][frame])))
            # TODO: Handle the errors better
            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not same lenght")
                continue

        mae = np.mean(errors)

        return round(mae, 2)
    
    def evaluate_relative_direction(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the accuracy of predicted directions compared to ground truth.
        """
        correct_predictions = 0
        total_predictions = len(filtered_frames["biome"])
        print(total_predictions)
        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            try:
                print(ground_truths[biome][trajectory][frame])
                print(predictions[biome][trajectory][frame])
                if ground_truths[biome][trajectory][frame] == predictions[biome][trajectory][frame]:
                    correct_predictions += 1
            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not same lenght")
                continue
        #total_predictions = len(filtered_frames["biome"])
        print(correct_predictions)
        accuracy = (correct_predictions / total_predictions) * 100
        return round(accuracy, 2)