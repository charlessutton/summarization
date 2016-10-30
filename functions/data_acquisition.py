import wikipedia

def rank_to_title(title, max_rank = 1):
    """
    For a given title, this function aims to get all the title of the related pages up to a given maximum rank.
    
    INPUTS 
    
    title : (str) an existing title name of a wikipedia page
    max_rank : the maximum rank to search links
    
   
    OUTPUT :
    rk_to_ttl : (dict) dictionnary of all the related pages with their rank to title
    """
    rk_to_ttl = {}
    page = wikipedia.page(title)
    rank = 0
    rk_to_ttl[title] = rank
    while rank < max_rank :
        rank += 1
        current_keys = rk_to_ttl.keys()
        for key in current_keys :
            if rk_to_ttl[key] == rank - 1 :
                links_of_this_key = wikipedia.page(key).links
                for link in links_of_this_key :
                    if link in current_keys :
                        pass
                    else :
                        rk_to_ttl[link] = rank                            
    return rk_to_ttl

def counter(rk_to_ttl): 
    """
    This function counts how much neighboors has a title from is rank_to_title dict.
    """
    counter = {}
    for key in rk_to_ttl.keys():
        value = rk_to_ttl[key]
        if value in counter.keys():
            counter[value]+=1
        else:
            counter[value] = 1
        
    return counter