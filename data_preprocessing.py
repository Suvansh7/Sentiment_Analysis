from icecream import ic

# Counter variable
iteration_counter = 0

def multilingual(a, translator):
    global iteration_counter  # Use the global counter variable
    
    try:
        iteration_counter += 1  # Increment the counter
        ic(iteration_counter) #
        return(translator.translate(a, dest="en").text)
        
    except:
        return(a)

def process(data, translator):
    data = data.head(50)
    data['new'] = data['Tweet'].apply(lambda review: multilingual(review, translator))
    return data
