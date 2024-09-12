import numpy as np
from PIL import Image

array = np.random.rand(3, 480, 768) * 255
array = array.astype(np.uint8)
array = array.transpose(1, 2, 0)

Image.fromarray(array).save("demo2.png")

def parse_action_vector(api_response):
    start = api_response.find('[')
    end = api_response.find(']')
    action_vec_str = api_response[start:end+1]

    action_vec_str = action_vec_str.lstrip('[').rstrip(']').replace(' ', '')
    action_vec = []
    for i in range(8):
        if i == 7:
            action = int(action_vec_str)
            action_vec.append(action)
        else:
            end_index = action_vec_str.find(',')
            action = int(action_vec_str[:end_index])   
            action_vec.append(action)
            action_vec_str = action_vec_str[end_index+1:]

    action_vec = np.array(action_vec)
    return action_vec

if __name__ == '__main__':
    #print(parse_action_vector("[1.2, 0, 0, 0, 0, 0, 0, 0]"))
    #print(type(parse_action_vector("[1, 0, 0, 0, 0, 0, 0, 0]")))
    action = parse_action_vector("[1, 0, 0, 0, 0, 0, 0, 0]")
    if isinstance(action, np.ndarray) and action.shape == (8,) and issubclass(action.dtype.type, np.integer):
        print("Valid action:", action)
    else:
        print("Invalid action predicted")