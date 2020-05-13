import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

os.chdir('.')
path = './data'

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = ('./data/'+ root.find('filename').text,
                     member[4][0].text,
                     member[4][1].text,
                     member[4][2].text,
                     member[4][3].text,
                     member[0].text
                     )
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax','class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df