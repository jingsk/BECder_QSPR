import numpy as np
from json import loads, dumps

def read_vasp_bec(xmlfile: str):
    """Given a vasprun.xml file location read and parse bec as np array with 
       indices(k, j, i): k over atoms, j over E-field x,y,z, i over forces along x,y,z
       Unit is |e| [(eV/A) / (V/A)]  
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()
    array_vectors = root.findall("./calculation/*[@name='born_charges']/set/v")
    bec = np.array([np.fromstring(element.text, sep=' ') for element in array_vectors])
    bec = bec.reshape(-1,3,3)
    return bec
    
def read_vasp_epsilon(xmlfile: str):
    """Given a vasprun.xml file location read and parse diel as np array 
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(xmlfile).getroot()
    array_vectors = root.findall("./calculation/*[@name='epsilon']/v")
    diel = np.array([np.fromstring(element.text, sep=' ') for element in array_vectors])
    return diel

#selection default to not break existing code
def ase_db_to_df(db_file, save=False):
    from ase.db import connect
    import pandas as pd
    db = connect(db_file)
    all_atoms = [row.toatoms() for row in db.select()]
    all_becder = [row.data.becder for row in db.select()]
    #all_diel = [row.data.diel for row in db.select(selection)]
    #energy = [row.energy for row in db.select()]
    #forces = [row.forces for row in db.select()]
    df = pd.DataFrame(data={'structure': all_atoms, 
                            'becder': all_becder,
                            #'energy': energy,
                            #'forces': forces,
                           },
                    )
    if save:
        df.to_csv('data.csv')
    return df


