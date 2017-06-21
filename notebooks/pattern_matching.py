import re
from  tweets_preprocess import *


def matched(line):
    # patterns for handling numbers
    digit_char_digit = re.compile(r"\d+([a-z]+)\d+")      # substitute with <alphanum>
    char_digit = re.compile(r"[a-z]+\d+")               # substitute with <alphanum>
    num = re.compile(r"\d+")                            # substitute with <num>


    line = preprocessTweets(line)
    # here we accumulate the result line after the processing
    result = []
    for word_index, word in enumerate(line.split()):
        if word[0] == '#':
            temp = re.sub(num, "<num>", word)
            result.append(temp)
            if temp != word:
                pass
                #print("{}\t\t-->\t\t{}".format(word, temp))
            continue

        # seach for pattern:   digits chars digits
        match_obj = re.match(digit_char_digit, word)
        if match_obj:       #if it matched
            if match_obj.group(1) == "x":     # for digits x digits we have special treatment e.g. 1366x768 --> <num>x<num>
                result.append("<num>x<num>")
                #print("{}\t\t-->\t\t{}".format(word, "<num>x<num>"))
            else:
                result.append("<alphanum>")
                #print("{}\t\t-->\t\t{}".format(word, "<alphanum>"))
        elif bool(re.match(char_digit, word)):
            result.append("<alphanum>")
            #print("{}\t\t-->\t\t{}".format(word, "<alphanum>"))
        else:
            searched_part = word
            result_chunk = ""

            temp = re.sub(num, "<num>", searched_part)
            result_chunk += temp
            result.append(result_chunk)


    return ' '.join(result)
