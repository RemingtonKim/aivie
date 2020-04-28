from poem import PoemGenerator
import json

#Generates 100 poems to a .txt file
if __name__ == '__main__':
    poem_gen = PoemGenerator()
    path = json.load(open('../data_lit/path.json', 'r'))['path']['OUT']
    out = open(path, 'w')
    for i in range(100):
        out.write(poem_gen.generate_poem())
        out.write('\n\n')
    out.close()