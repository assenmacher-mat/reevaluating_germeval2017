#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

### function to create pre-processed dataframe used for subtask A and B --> outer_df ###
def create_outer_df(root):
    """ 
    create dataframe to work with out of outer tree (=extract only informations on document level).

    Arg: root: created out of XML file using tree.getroot()
    """

    columns = ["id", "text", "relevance", "sentiment", "opinion"]
    df = pd.DataFrame(columns = columns)
    rowlst = []
    for node in root:
        id = node.attrib.get("id")
        text = node.find("text").text
        relevance = node.find("relevance").text if node is not None else np.nan
        sentiment = node.find("sentiment").text if node is not None else np.nan
        opinion = node.find("Opinions") is not None # extracts flag for opinion
        #df = df.append(pd.Series([id, text, relevance, sentiment, opinion], index = columns), ignore_index = True)
        rowlst.append(pd.Series([id, text, relevance, sentiment, opinion], index = columns))
    df_extended = pd.DataFrame(rowlst, columns=columns)
    out = pd.concat([df, df_extended])
    out.index = range(len(out))
    return out

### functions to create pre-processed dataframe used for subtask C and D --> df ###
def iter_opinions(document):
    """ 
    iterate over opinions in a document. Return dictionary with subnode entries.

    Arg: 
      document: part of tree (subnode) extracted from XML file
    """
    document_id = document.attrib.get("id")
    document_text = document.find("text").text
    document_relevance = document.find("relevance").text
    document_sentiment = document.find("sentiment").text
    document_attr = {"id": document_id, "text": document_text, "relevance": document_relevance, "sentiment": document_sentiment}

    if document.find('Opinions') is None:
        opinion_dict = document_attr.copy()
        opinion_dict.update({"category":np.nan, "from":np.nan, "to":np.nan, "target":np.nan, "polarity":np.nan})
        yield opinion_dict
    else:
        for opinion in document.iter('Opinion'):
            opinion_dict = document_attr.copy()
            opinion_dict.update(opinion.attrib)
            yield opinion_dict

def iter_docs(tree):
    """
    iterate over documents in tree.

    Arg: 
      tree: a tree extracted from .xml file (using ElementTree)
    """
    for document in tree.iter('Document'):
        for row in iter_opinions(document):
            yield row

def convert_df(tree):
    """
    convert tree into dataframe, add aspect variable and correct data entries.

    Arg: 
      tree: a tree extracted from .xml file (using ElementTree)
    """
    df = pd.DataFrame(list(iter_docs(tree)))
    # add aspect variable
    aspect = df.category.str.split('#').str[0]
    df["aspect"] = aspect
    # replace wrongly written entries for polarity
    df.polarity = df.polarity.replace('positve', 'positive') # comes with train_df
    df.polarity = df.polarity.replace(' negative', 'negative') # comes with test_dia_df
    # create aspect_polarity
    df['aspect_polarity'] = [':'.join([str(x), str(y)]) for x, y in zip(df['aspect'], df['polarity'])]
    return df

### additional load and prepare data functions for subtask C ###
def preproc_subtaskC(df, df_outer, cats, part_task):
    """ 
    pre-process data for subtask C1 if part_task == "aspect" and 
    for subtask C2 if part_task == "aspect_polarity".

    Args:
        df: dataframe
        outer_df: outer dataframe
        cats: categories (as list)
        part_task: "aspect" or "aspect_polarity"; defining the part of subtask C
    """
    # create nullmatrix
    matrix = np.zeros((len(df),len(cats)))
    opinions_index = pd.DataFrame(matrix, columns = cats)
    df_long = pd.concat([df, opinions_index], axis = 1)

    # fill opinions_index
    for i in np.arange(0, len(df_long)):
        for j in cats:
            if df_long[part_task][i] == j:
                df_long.loc[i,j] = 1
    # aggregate by id
    aggregations = {i:sum for i in cats}
    df_agg = df_long.groupby('id').agg(aggregations)
    # merge df_agg to df_outer (df without opinions)
    df_outer = pd.merge(df_outer, df_agg, on='id', how='left')
    # convert to 0 / 1 labels
    for j in cats:
        df_outer[j] = df_outer[j].astype(bool)
        df_outer[j] = df_outer[j].astype(int)
        for i in np.arange(0, len(df_outer)):
            if df_outer.loc[i,j] > 0:
                df_outer.loc[i,j] = 1
    # delete irrelevant rows
    df_outer = df_outer.loc[df_outer.opinion, :]
    return df_outer

def get_cats(df_path, xml_filename = "train-2017-09-15.xml", part_task = "aspect"):
    """
    return full list of aspect/aspect+polarity categories as list.

    Args:
        df_path: path to data folder
        xml_filename: file name of XML dataset. Default is XML train dataset.
        part_task: "aspect" or "aspect_polarity"; defining part of Subtask C. 
    """
    tree = ET.parse(df_path+xml_filename)
    df = convert_df(tree)
    cats = []
    if part_task == "aspect":
        cats = df.aspect.unique()
        cats = np.delete(cats, 2) # delete nan
        cats = np.append(cats,'QR-Code')

    if part_task == "aspect_polarity":
        cats = df.aspect_polarity.unique()
        # add Gepäck:positive and all QR_Code combinations
        add = ['Gepäck:positive', 'QR-Code:negative', 'QR-Code:neutral', 'QR-Code:positive']
        cats = sorted(np.append(cats, add))
        # delete nan:nan
        cats = np.delete(cats, -1)
    return cats

### load and prepare functions for subtask D --> convert "from" and "to" sequence indices to BIO tags ###
def prep_df(df):
    """
    prepare data for subtask D.

    Arg:
        df: pre-processed dataframe
    """
    # drop NAs in opinion
    pdf = df.dropna(subset = ["target"])
    pdf = pdf[pdf.target != "NULL"]
    pdf[['from', 'to']] = pdf[['from', 'to']].astype(int)
    # if from > to, switch positions
    pdf.loc[pdf['to'] < pdf['from'], ['from', 'to']] = pdf.loc[pdf['to'] < pdf['from'], ['to', 'from']].values

    # create labels
    # define dictionary for categories and polarity
    aspect_dict = {
        'Allgemein': 'ALG', 
        'Atmosphäre': 'ATM', 
        'Informationen': 'INF',
        'DB_App_und_Website': 'APP',
        'Auslastung_und_Platzangebot': 'AUP',
        'Sonstige_Unregelmässigkeiten': 'SOU',
        'Zugfahrt': 'ZUG', 
        'Ticketkauf': 'TIC', 
        'Sicherheit': 'SIC',
        'Barrierefreiheit': 'BAF', 
        'Service_und_Kundenbetreuung': 'SUK', 
        'Connectivity': 'CON', 
        'Komfort_und_Ausstattung': 'KUA',
        'Toiletten': 'TOI', 
        'Gastronomisches_Angebot': 'GST', 
        'Image': 'IMG', 
        'Design': 'DSG', 
        'Reisen_mit_Kindern': 'RMK',
        'Gepäck': 'GEP',
        'QR-Code': 'QRC'
    }
    polarity_dict = {
        'neutral': 'NEU',
        'positive': 'POS',
        'negative': 'NEG'
    }
    pdf = pdf.replace({"aspect":aspect_dict, "polarity":polarity_dict})
    entities = pdf.aspect + ":" + pdf.polarity    
    pdf[['from', 'to']] = pdf[['from', 'to']].astype(int)
    return pdf, list(set(entities))

def transform_df(df):
    '''
    transform prepared data i.e. create dictionary of dataentry
    {"text", "start", "end", "target", "asp_pol"}

    Args: 
        df: dataframe
    '''
    tdf = []
    for i in df.id.unique():
        subset = df[df.id == i]
        entry = {"text": subset.text.iloc[0], 
                 "start": [start for start in subset['from']], 
                 "end": [end for end in subset['to']], 
                 "target": [tar for tar in subset['target']], 
                 "asp_pol": [asp + ":" + pol for asp, pol in zip(subset['aspect'], subset['polarity'])]}
        tdf.append(entry)
    return tdf

def remove_O(O_list):
    """ 
    'O' handling in BIO_tags. Remove 'O' if word has tag and unlist ['O'].

    Args:
        O_list: list with BIO_tags
    """
    if len(O_list) > 1 and 'O' in O_list: 
        O_list.remove("O")
    #if len(O_list) == 1:
        #O_list = O_list[0]
    return O_list

def split_text(text, indices):
    """
    split text first by target positions, second by spaces.

    Args:
        text: string
        indices: list of start and end positions of targets
    """
    #remove duplicates
    indices = [*set(indices)]
    indices.sort()
    # split text
    parts = [text[i:j] for i,j in zip(indices, indices[1:]+[None])]
    # concatenate with spaces and split again
    parts = " ".join(parts).split()
    return parts

def bio_tagging_sentence(dataentry):
    """
    add BIO tags to documents.

    Arg:
        dataentry: prepared dataframe entry as tuple of text and dictionary (see transform_df())
    """
    indices = [0] + dataentry['start'] + dataentry['end']
    dataentry['target_pos'] = [dataentry['text'][j:k] for j, k in zip(dataentry['start'], dataentry['end'])] # target based on positions
    df0 = pd.DataFrame(dataentry)
    df0 = df0.drop(columns=['start', 'end', 'target'])
    df0['text'] = df0.apply(lambda row: split_text(row['text'], indices), axis=1)
    df1 = (df0.set_index(['target_pos','asp_pol'], append=True)
          .explode('text')
          .stack()
          .reset_index(level=3, drop=True)
          .reset_index(name='Word')
          .rename(columns={'level_0':'Sentence'})
          )
    #find target by pattern
    pattern = [tar.split() for tar in dataentry['target_pos']]
    B_index = [[df1.index[i - len(pat)] # Get the index 
                for i in range(len(pat), len(df1)+1)
                if all(df1['Word'][i-len(pat):i] == (pat))] # if it is pattern
                for pat in pattern] # for each pattern
    # handle duplicates
    pattern_pol = [a + b for a, b in zip(dataentry['target_pos'], dataentry['asp_pol'])]
    for tar in set(pattern_pol):
        dups = [i for i, x in enumerate(pattern_pol) if x == tar] # same target and entity -> not same pos
        pat_count = len(dups)
        if pat_count > 1:
            for j, dup in enumerate(dups):
                B_index[dup] = B_index[dup][j::pat_count]                
    B_index = [B_index[j][j] for j in range(0, len(pattern))]
    I_index = [B_index[i] + j for i in range(0, len(B_index)) for j in range(1, len(pattern[i])) if len(pattern[i]) > 1]
    
    # get entities
    entities = [(ent, int(start/(n+1)), int((start/(n+1))+len(pattern[n])-1)) 
                for n, (ent, start) in enumerate(zip(dataentry['asp_pol'], B_index))]
    # assign bio tags
    splitted = df1['target_pos'].str.split()
    e = df1.pop('asp_pol')
    m1 = pd.Series([i in B_index for i in set(list(df1.index))]) # beginning of pattern
    m2 = [i in I_index for i in set(list(df1.index))]
    m3 = [b in a for a, b in zip(splitted, df1['Word'])] # where '*' in a
    df1['asp_pol'] = np.select([m1, m2 & ~m1 & m3], ['B-' + e, 'I-' + e],  default='O')

    # handle multi-labeling
    ent_cols = ['asp_pol0']
    subset = df1[df1.Sentence == 0].rename(columns = {'asp_pol':'asp_pol0'})
    for i in df1.Sentence.unique():
        if i == 0:
            next
        subset['asp_pol' + str(i)] = df1[df1['Sentence'] == i].asp_pol.values
        ent_cols.append('asp_pol' + str(i))
    subset['asp_pol'] = subset[ent_cols].apply(lambda x: remove_O(list(set(x))), axis = 1)
    subset.drop(ent_cols + ['Sentence', 'target_pos'], axis = 1, inplace = True)
    words = list(subset['Word'])
    bio_tags = list(subset['asp_pol'])
    return words, bio_tags, entities

def bio_tagging_df(df, df_type = None):  
    """
    Return complete pre-processed dataframe of text and corresponding BIO tags.
    Handle exceptions manually, depending on dataframe.

    Args:
        df: dataframe
        df_type: 'train_df', 'dev_df', 'test_syn_df', 'test_dia_df' or None
    """
    pdf, _ = prep_df(df)
    tdf = transform_df(pdf)

    # exception handling
    if df_type == "train_df":
        for i in ['start', 'end', 'target', 'asp_pol']:
            tdf[549][i].pop(1)
            tdf[612][i].pop(2)
            tdf[763][i].pop(4)
            tdf[1991][i].pop(1)
            tdf[2629][i].pop(2)
            tdf[2723][i].pop(1)  
            tdf[4887][i].pop(2)
            tdf[5948][i].pop(1)
            tdf[6199][i].pop(1)
            tdf[6336][i].pop(1)
            tdf[6394][i].pop(6)
        
        tdf[307]['text'] = tdf[307]['text'][:81] + ' ' + tdf[307]['text'][81:] # buchen, -> buchen ,
        tdf[307]['end'][1] = 112

        tdf[1486]['text'] = tdf[1486]['text'][:76] + ' ' + tdf[1486]['text'][76:] # aus, -> aus ,
        tdf[1486]['end'][0] = 104

        tdf[1985]['text'] = tdf[1985]['text'][:52] + ' ' + tdf[1985]['text'][52:] # aus, -> aus ,
        tdf[1985]['start'][2] = 105
        tdf[1985]['end'][1] = 82
        tdf[1985]['end'][2] = 122

        # abgeschnittenes Wort 'icherheit-' aus target raus
        tdf[3054]['start'][1] = 71
        tdf[3054]['target'][1] = 'ich ignoriere die Bahn'
        
        #correct start and end values
        tdf[3157]['start'][1] = 123
        tdf[3157]['end'][1] = 129

        tdf[3460]['text'] = tdf[3460]['text'][:204] + ' ' + tdf[3460]['text'][204:] # Senioren, -> Senioren ,
        tdf[3460]['start'][2:] = [207, 313, 463]
        tdf[3460]['end'][2:] = [223, 354, 489]
        tdf[3460]['end'][0] = 348

        tdf[3873]['text'] = tdf[3873]['text'][:54] + ' ' + tdf[3873]['text'][54:] # pfiff, -> pfiff ,
        tdf[3873]['end'][1] = 82

        tdf[4038]['text'] = tdf[4038]['text'][:50] + ' ' + tdf[4038]['text'][50:60] + ' ' + tdf[4038]['text'][60:]
        tdf[4038]['end'][2:] = [61, 83]

        tdf[4122]['start'][5] = 158 # wrong pos values, did not correspond to target
        tdf[4122]['end'][5] = 185

        tdf[4150]['text'] = tdf[4150]['text'][:112] + " " + tdf[4150]['text'][112:] # /Dreckige -> / Dreckige
        tdf[4150]['start'][4:] = [140, 273]
        tdf[4150]['end'][2:] = [130, 130, 151, 276]

        tdf[4341]['start'][3] = 89 # wrong pos values, did not correspond to target
        tdf[4341]['end'][3] = 125

        tdf[4416]['start'][2] = 291
        tdf[4416]['end'][2] = 319

        tdf[5243]['start'][1] = 59
        tdf[5243]['end'][1] = 72

        tdf[5424]['end'][2] = 391 # correct only end of target

        tdf[5516]['text'] = tdf[5516]['text'][:519] + ' ' + tdf[5516]['text'][519:] # lassen, -> lassen ,
        tdf[5516]['start'][4] = 596
        tdf[5516]['end'][2:] = [540, 540, 605]

        tdf[5719]['text'] = tdf[5719]['text'][:51] + ' ' + tdf[5719]['text'][51:]
        tdf[5719]['start'][2:] = [68, 156, 189, 355]
        tdf[5719]['end'][1:] = [62, 108, 165, 202, 368]

        tdf[5883]['start'][3] = 105
        tdf[5883]['end'][3] = 126

        tdf[6240]['start'][1] = 22 # correct pos of target
        tdf[6240]['end'][1] = 34

        tdf[6276]['start'][2] = 447
        tdf[6276]['end'][2] = 492

        tdf[6277]['start'][3] = 157
        tdf[6277]['end'][3] = 180
    
    if df_type == "dev_df":
        # remove part-of-word tags where whole word is already tagged with the same entity
        for i in ['start', 'end', 'target', 'asp_pol']:
            tdf[221][i].pop(4)
            tdf[441][i].pop(1)
        tdf[612]['end'][0] = 67 # take whole 'word': mysteriöser -> mysteriöser...
        tdf[643]['text'] = tdf[643]['text'][:39] + ' ' + tdf[643]['text'][39:] # 'schneller...Wenn' -> 'schneller ...Wenn'
        tdf[643]['end'][1] = 61
    
    if df_type == "test_syn_df":
        for i in ['start', 'end', 'target', 'asp_pol']:
            tdf[366][i].pop(1) 
            tdf[429][i].pop(2)
            tdf[618][i].pop(2) # exact double mention
            tdf[750][i].pop(2)
        tdf[432]['start'][1] = 112 # correct start of target
    
    if df_type == "test_dia_df":
        for i in ['start', 'end', 'target', 'asp_pol']:
            tdf[433][i].pop(1) # exact double mention
        # correct entry position
        tdf[470]['start'][3] = 327
        tdf[470]['end'][3] = 335
        tdf[470]['target'][3] = 'Toilette'

    bio_tags_df = [bio_tagging_sentence(l) for l in tdf]
    bio_tags_df = pd.DataFrame(bio_tags_df, columns = ["text", "bio_tags", "entity"])
    return bio_tags_df


def sample_to_tsv(df_path, xml_filename, save_as_tsv=True):
    """
    pre-process data and save as TSV file.

    Args:
      df_path: location of dataframe
      xml_filename: name of file in XML format
      save_as_tsv: flag for saving as TSV file
    """
    
    print("Original XML Dateframe: ", xml_filename)
    tree = ET.parse(df_path+xml_filename)
    root = tree.getroot()

    ################## data for subtask A + B ########################
    df = create_outer_df(root)
    
    ################## data for subtask C + D ########################
    df_op = convert_df(tree)
    df_op = df_op.dropna(subset = ["text"])    
    
    ################## data for subtask C ############################
    # create categories array (same for all data!)
    cats = get_cats(df_path, part_task = "aspect")
    df_cat = preproc_subtaskC(df_op, df, cats, 'aspect')

    cats_pol = get_cats(df_path, part_task = "aspect_polarity")
    df_cat_pol = preproc_subtaskC(df_op, df, cats_pol, 'aspect_polarity')

    if save_as_tsv:
        df_type = xml_filename.split("-")[0]
        # for subtask A + B
        df.to_csv(df_path+df_type+"_df.tsv", sep="\t", index = False, header = True)
        # for subtask C
        df_cat.to_csv(df_path+df_type+"_df_cat.tsv", sep="\t", index = False, header = True)
        df_cat_pol.to_csv(df_path+df_type+"_df_cat_pol.tsv", sep="\t", index = False, header = True)
        # for subtask D (without BIO tags)
        df_op.to_csv(df_path+df_type+"_df_opinion.tsv", sep="\t", index = False, header = True)


def main():
    """
    pre-process and save full data: train, dev, test_syn, test_dia.
    will save 16 (4x4) TSV files in data folder (see sample_to_tsv()).
    """

    df_path = "./data/"
    sample_to_tsv(df_path, "train-2017-09-15.xml")
    sample_to_tsv(df_path, "dev-2017-09-15.xml")
    sample_to_tsv(df_path, "test_syn-2017-09-15.xml")
    sample_to_tsv(df_path, "test_dia-2017-09-15.xml")
    print("Complete data is saved as TSV files in ", df_path)

if __name__ == "__main__":
    main()