import os
from os.path import join


if __name__ == '__main__':
    cwd = os.getcwd()
    chunks_dir = join(cwd, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    line_counter = 1
    file_counter = 1
    chunk_lines = []

    with open("article_titles_pre_filtered.txt", "r", encoding="utf-8") as file:
        for line in file:
            chunk_lines.append(line)
            if line_counter == 50:
                with open(join(chunks_dir, f"titles_chunk_{file_counter}.txt"), "w", encoding="utf-8") as file_2:
                    file_2.writelines(chunk_lines)
                # Reset for the next chunk
                chunk_lines = []
                line_counter = 0
                file_counter += 1
            line_counter += 1

    # Write any remaining lines to a final chunk file if less than 50
    if chunk_lines:
        with open(join(chunks_dir, f"titles_chunk_{file_counter}.txt"), "w", encoding="utf-8") as file_2:
            file_2.writelines(chunk_lines)