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
        single_errors = {}

        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            biome, trajectory, frame = str(biome), str(trajectory), str(frame)
            
            try:
                ae = np.abs(np.array(ground_truths[biome][trajectory][frame][0]) - np.array(predictions[biome][trajectory][frame]))
                errors.append(ae)

                if biome not in single_errors:
                    single_errors[biome] = {}
                if trajectory not in single_errors[biome]:
                    single_errors[biome][trajectory] = {}
                
                single_error = ae / np.array(ground_truths[biome][trajectory][frame][0])
                single_errors[biome][trajectory][frame] = round(single_error, 2)
            
            # TODO: Handle the errors better
            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not same lenght")
                continue

        mae = np.mean(errors)

        return round(mae, 2), single_errors
    
    def evaluate_relative_distance(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the Mean Absolute Error (MAE) between predictions and ground truths
        """
        errors = []
        single_errors = {}

        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            biome, trajectory, frame = str(biome), str(trajectory), str(frame)
            
            try:
                ae = np.abs(np.array(ground_truths[biome][trajectory][frame]) - np.array(predictions[biome][trajectory][frame]))
                errors.append(ae)

                if biome not in single_errors:
                    single_errors[biome] = {}
                if trajectory not in single_errors[biome]:
                    single_errors[biome][trajectory] = {}

                single_error = ae / np.array(ground_truths[biome][trajectory][frame])
                single_errors[biome][trajectory][frame] = round(single_error, 2)

            # TODO: Handle the errors better
            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not same lenght")
                continue

        mae = np.mean(errors)

        return round(mae, 2), single_errors
    
    def evaluate_relative_direction(self, filtered_frames, predictions, ground_truths):
        """
        Calculates the accuracy of predicted directions compared to ground truth.
        """
        single_errors = {}
        correct_predictions = [0, 0, 0]
        total_predictions = len(filtered_frames["biome"])
        for biome, trajectory, frame in zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"]):
            biome, trajectory, frame = str(biome), str(trajectory), str(frame)
            try:
                for dim in range(3):
                    if ground_truths[biome][trajectory][frame][dim] == predictions[biome][trajectory][frame][dim]:
                        correct_predictions[dim] += 1

            except Exception as e:
                print(e)
                print("None encountered in predictions or Ground Truths and Predictions are not the same lenght")
                continue

        accuracy = [round((correct_predictions[i] / total_predictions) * 100, 2) for i in range(3)]
        
        return accuracy, single_errors