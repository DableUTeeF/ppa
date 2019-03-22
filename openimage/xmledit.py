from xml.etree import cElementTree as ET
import os


if __name__ == '__main__':
    path = '/media/palm/data/ppa/v6/anns/val'
    spath = '/media/palm/data/ppa/v6/anns/val'
    for file in os.listdir(path):
        tree = ET.parse(os.path.join(path, file))
        for elem in tree.iter():
            if 'name' in elem.tag:
                # if elem.text.lower() == 'vehicle registration plate':
                #     elem.text = 'LP'
                if elem.text == 'Person':
                    elem.text = 'person'
                if elem.text == 'goodshoe':
                    elem.text = 'goodshoes'
                if elem.text == 'badshoe':
                    elem.text = 'badshoes'
                if elem.text == 'Helmet':
                    elem.text = 'goodhelmet'
                # if elem.text == 'shoes':
                #     elem.text = 'goodshoes'
                # if elem.text == 'Boot':
                #     elem.text = 'goodshoes'
                if elem.text == 'unsafe_hat':
                    elem.text = 'badhelmet'
                if elem.text == 'helmet':
                    elem.text = 'goodhelmet'
                if elem.text == 'good':
                    elem.text = 'person'
                if elem.text == 'personwithouthat':
                    elem.text = 'person'
                if elem.text == 'bad':
                    elem.text = 'person'
                if elem.text == 'noshoes':
                    elem.text = 'person'
                if elem.text == 'nohelmet':
                    elem.text = 'person'
                # if elem.text == 'person':
                #     elem.text = 'badperson'
                # if elem.text == 'unsafe_person':
                #     elem.text = 'person'
                # if elem.text == 'hat':
                #     elem.text = 'helmet'
        tree.write(os.path.join(spath, file))
