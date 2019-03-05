from xml.etree import cElementTree as ET
import os


if __name__ == '__main__':
    path = '/media/palm/data/ppa/v3/anns/train'
    for file in os.listdir(path):
        tree = ET.parse(os.path.join(path, file))
        for elem in tree.iter():
            if 'name' in elem.tag:
                # if elem.text.lower() == 'vehicle registration plate':
                #     elem.text = 'LP'
                # if elem.text == 'Person':
                #     elem.text = 'person'
                # if elem.text == 'Helmet':
                #     elem.text = 'helmet'
                # if elem.text == 'shoes':
                #     elem.text = 'goodshoes'
                # if elem.text == 'unsafe_shoes':
                #     elem.text = 'badshoes'
                # if elem.text == 'safe_person':
                #     elem.text = 'person'
                if elem.text == 'person_with_hat':
                    elem.text = 'person'
                if elem.text == 'unsafe_person':
                    elem.text = 'person'
                # if elem.text == 'hat':
                #     elem.text = 'helmet'
        tree.write(os.path.join(path, file))
