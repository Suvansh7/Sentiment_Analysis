from icecream import ic
def multilingual(a,translator):
    try:
        ic(100)
        return(translator.translate(a, dest="en").text)
        
    except:
        
        return(a)

def process(data,translator):
    data['new'] = data['Tweet'].apply(lambda review: multilingual(review,translator))
    return(data)

