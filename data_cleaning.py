def clean(data):
    
    data.dropna(inplace=True)
    empty=[]
    for i,j,date,user,txt,likes,retweets in data.itertuples():
        if(type(txt)==str):
            if(txt.isspace()):
                empty.append(i)
    data.drop(empty, inplace=True)
    data = data.drop_duplicates(subset=None,keep="first")
    return(data)
    
    