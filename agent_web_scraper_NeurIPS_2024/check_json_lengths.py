import json

if __name__ == '__main__':
    request_again = []
    for i in range(1, 77):
        with open(f"gpt_filtered/gpt_response_{i}.json", "r", encoding="utf-8") as f:
            try:
                gpt_response = json.load(f)
                #print(f"***********{i}**********")
                if len(gpt_response) != 50:
                    print(f"File: gpt_response_{i} Length: {len(gpt_response)}")
                    request_again.append(i)
            except Exception as e:
                print(f"Error in {i}: {e}")
                request_again.append(i)
    print(request_again)
