'''
# this is the code needed for running the sentiment analysis engine
import pandas as pd
import re
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
#from textblob import TextBlob,
from textblob import Word, Blobber
import textblob_fr
import textblob_de
import os
import shutil # from removing files from one folder to another
os.environ["NLS_LANG"] = ".AL32UTF8" 
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
input_directory='/home/ayaqubov/engines/new_engine/DriverShare/Input/'
files_in_folder=os.listdir(input_directory)
#for i in range()
#files_in_folder

output_directory='/home/ayaqubov/engines/new_engine/DriverShare/Output/'
processed_directory='/home/ayaqubov/engines/new_engine/DriverShare/Processed/'


# check while the input is empty do the 

#while(os.listdir('/home/ayaqubov/engines/Windows-share/Input/')!=""):
# get the data and process the data

for ifiles in range(0,len(files_in_folder)):
    file_directory=input_directory + files_in_folder[ifiles]
    print(file_directory)
    #file_directory
    mycolumnsdf=['REVIEW_WID','REVIEW_FULL_TEXT']
    df_aws_reviews=pd.DataFrame(columns=mycolumnsdf)
    df_aws_reviews=pd.read_csv(file_directory)
    df_aws_reviews['LANGUAGE']=""
    num_unique_reviews=df_aws_reviews.shape[0]

    ###########################################################
    # here comes the check the language part and write correspondingly to the df_aws_reviews dataframe

    from nltk.tokenize import sent_tokenize, word_tokenize
    ## data processing
    print('Processing...')
    index=0
    badcounter=0
    for u in range(0,num_unique_reviews):
        review_id=df_aws_reviews.iloc[u,0]
        print(review_id)
        review_text=df_aws_reviews.iloc[u,1]
        review_str=str(review_text)
        #review_strw=review_str.encode('cp1252')## windows encoded
        #ureview_str=review_strw.decode('utf-8')
        #ureview_str=unicode(review_str,"utf-8")
        try:
            #ureview_str=review_str.decode('cp1252').encode('utf-8',errors='ignore')
            ureview_str=review_str.encode('utf-8',errors='ignore')
            ### windows does not support utf-8, keep in mind that instead it uses utf-16.
            #### Keep this in mind when using documents created on a Windows machine or saving files in a Windows machine.

            #print(review_str)
            #print(len(review_str))
            contain_l=re.search('[a-zA-Z]', review_str)
            if(contain_l!='None'):
                # handle japanese,arabic,chinese cases because they appear as the ?? marks in the results
                try:
                    text_lang=detect(ureview_str)
                    #text_lang2=identify_lang(review_str)
                except:
                    print 'Error in reading, keep reading'
                    #i_debug+=1
                    continue
                #check_words=word_tokenize(review_str)
                #for iiii in range(0,len(check_the_words)):
                #    if (check_the_words[iiii]=='product' or check_the_words[iiii]=='excellent'  or check_the_words[iiii]=='mouse'):
                #        break
                #            #continue

                if(text_lang=='en'):
                    df_aws_reviews.iloc[u,2]='English'
                    index+=1

                if(text_lang=='de'):
                    df_aws_reviews.iloc[u,2]='German'
                    index+=1

                if(text_lang=='fr'):
                    if('product' in ureview_str or 'excellent' in ureview_str or 'mouse' in ureview_str):
                        df_aws_reviews.iloc[u,2]='English'
                    else:
                        df_aws_reviews.iloc[u,2]='French'
                    index+=1
        except:
            # windows encoded file is handled seperately
            ureview_str=review_str.decode('cp1252').encode('utf-8',errors='ignore')
            #ureview_str=review_str.encode('utf-8',errors='ignore')
            ### windows does not support utf-8, keep in mind that instead it uses utf-16.
            #### Keep this in mind when using documents created on a Windows machine or saving files in a Windows machine.

            #print(review_str)
            #print(len(review_str))
            contain_l=re.search('[a-zA-Z]', review_str)
            if(contain_l!='None'):
                # handle japanese,arabic,chinese cases because they appear as the ?? marks in the results
                try:
                    text_lang=detect(ureview_str)
                    #text_lang2=identify_lang(review_str)
                except:
                    print 'Error in reading, keep reading'
                    #i_debug+=1
                    continue
                #check_words=word_tokenize(review_str)
                #for iiii in range(0,len(check_the_words)):
                #    if (check_the_words[iiii]=='product' or check_the_words[iiii]=='excellent'  or check_the_words[iiii]=='mouse'):
                #        break
                #            #continue

                if(text_lang=='en'):
                    df_aws_reviews.iloc[u,2]='English'
                    index+=1

                if(text_lang=='de'):
                    df_aws_reviews.iloc[u,2]='German'
                    index+=1

                if(text_lang=='fr'):
                    if('product' in ureview_str or 'excellent' in ureview_str or 'mouse' in ureview_str):
                        df_aws_reviews.iloc[u,2]='English'
                    else:
                        df_aws_reviews.iloc[u,2]='French'
                    index+=1
            print('Error happened in character decoding process::UnicodeDecodeError')
            badcounter+=1
        print('Counter bad unicodes are :: ', badcounter)

    # delete reviews that does not have language information
    # can also use the drop function from the pandas library
    df_aws_reviews=df_aws_reviews[df_aws_reviews['LANGUAGE'] != ""]
    df_aws_reviews.shape

    #functions to process the data:
    def count_words(sentence):
        words=word_tokenize(sentence)
        l=len(words)
        return l

    def get_sentiment_en(sentence):
        from textblob import TextBlob
        # this function is the simplest function
        blob=TextBlob(sentence)
        sentiment_pol=blob.sentiment.polarity
        #sentiment_sub=TextBlob(sentence).sentiment.subjectivity
        #sentiment_=sentiment_pol
        return sentiment_pol
    from textblob_fr import PatternTagger, PatternAnalyzer
    from textblob import Blobber
    def get_sentiment_fr(sentence):
        mysentence=str(sentence)
        tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        sentim = tb(mysentence).sentiment
        sentiment_pol=sentim[0]
        sentiment_sub=sentim[1]
        return sentiment_pol

    from textblob_de import TextBlobDE as TextBlob
    from textblob_de import PatternTagger
    def get_sentiment_de(sentence):
        blob = TextBlob(sentence)
        sentim=blob.sentiment
        sentiment_pol=sentim[0]
        sentiment_sub=sentim[1]
        return sentiment_pol
    import langid
    def identify_lang(sentence):
        cl=langid.classify(sentence)
        lan=cl[0]
        return lan
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    ps = PorterStemmer()
    def stem_word(in_word):
        stemmed_word=ps.stem(in_word)
        return stemmed_word




    #########################################################
    # division into the sentences
    df_num_of_rows=df_aws_reviews.shape[0]
    #mycolumns=['REVIEWSENTENCE_WID','REVIEW_WID','SENTENCE_ID','SKU','COUNTRY','SITE_URL','REVIEW_POSTED_DATE','WORD_COUNT',
    #                    'SENTIMENT','STAR_RATING','SENTENCE','PRODUCT_TYPE','PRODUCT_GROUP','PRODUCT_LINE_NAME']
    mycolumns_sentence=['textsentence_id','text_id','sentence_id','word_count',
                        'sentiment','sentence','language']
    df_aws_sentences=pd.DataFrame(columns=mycolumns_sentence)

    # adding sentences
    index_=0
    for i in range(0,df_num_of_rows):
        this_review=df_aws_reviews.iloc[i,1]
            # use try except because errpr was occuring in some cases
        try:
            sentences_this_review=sent_tokenize(this_review)
        except:
            print("Error in sentence tokenizing")
            continue
        num_of_sents=len(sentences_this_review)
        current_review_id=str(df_aws_reviews.iloc[i,0])
        #print(current_review_id)
        if(num_of_sents!=0):
            sent_id=0
            for j in range(0,num_of_sents):
                current_sentence=sentences_this_review[j]
                if(current_sentence in ["!","?","."]):
                    continue
                word_count=count_words(current_sentence)
                reviewsentence_id=current_review_id+'_'+str(sent_id)#int(current_review_id+'_'+str(sent_id))
                # Now calculate the polarity of sentece:
                #sentiment_=get_sentiment(current_sentence)
                if(df_aws_reviews.iloc[i,2]=='English'):
                    sentiment_=get_sentiment_en(current_sentence)
                elif(df_aws_reviews.iloc[i,2]=='French'):
                    sentiment_=get_sentiment_fr(current_sentence)
                elif(df_aws_reviews.iloc[i,2]=='German'):
                    sentiment_=get_sentiment_de(current_sentence)

                one_row=[reviewsentence_id,current_review_id,sent_id,word_count,sentiment_,current_sentence,df_aws_reviews.iloc[i,2]]
                df_aws_sentences.loc[index_]=one_row
                sent_id+=1
                index_+=1

    df_aws_sentences[['sentence_id','word_count']]=df_aws_sentences[['sentence_id','word_count']].astype(int)
    ########################################################
    ## now word frequency table

    cols_word_freq= ['reviewsentence_wid','review_wid','sentence_id','word','translated_word','freq']
    df_sents_num_of_rows=df_aws_sentences.shape[0]


    df_word_freq=pd.DataFrame(columns=cols_word_freq)
    # get rid of commas etc
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    import re
    def check_num(input_s): 
        num_format = re.compile("^[\-]?[1-9][0-9]*\.?[0-9]+$")
        isnumber = re.match(num_format,input_s)
        #isnumber=~
        if isnumber:
            return True
        else:
            return False
    def check_letter(input_s):
        #remove len 1 and 2s(come back for len 2 later)
        l=len(input_s)
        if(l==1 or l==2):
            return True
        return False

    from nltk.corpus import stopwords
    #stopwords_ = set(stopwords.words('english'))
    stopwords_fr=set(stopwords.words('french'))
    stopwords_en=set(stopwords.words('english'))
    stopwords_ge=set(stopwords.words('german'))


    windex_=0

    for i in range(0,df_sents_num_of_rows):
        #print(i)
        sentence=df_aws_sentences.iloc[i,5]
        #words=word_tokenize(sentence)
        # maybe use try-except block as follows:
        #try:
        #words=tokenizer.tokenize(sentence)
        #except:
        #print("error in word tokenizing")
        words=tokenizer.tokenize(sentence)
        tags_=nltk.pos_tag(words)
        num_words=len(words)
        for j in range(0,num_words):
            word=words[j]
            wordlow=word.lower()
            # check if it is noun here
            translated=wordlow  ## for another language we need translation
            freq=1
            w_isnum=check_num(wordlow)
            one_two_let=check_letter(wordlow)
            if(df_aws_sentences.iloc[i,6]=='English'):
                stopwords_=stopwords_en
            if(df_aws_sentences.iloc[i,6]=='French'):
                stopwords_=stopwords_fr
            if(df_aws_sentences.iloc[i,6]=='German'):
                stopwords_=stopwords_ge
            if(wordlow in stopwords_ or w_isnum or one_two_let):
                continue
'''

'''

            if(tags_[j][1]=='NN' or tags_[j][1]=='NNS' or tags_[j][1]=='NNP'):
                # since we expect aspects are more likely to be among the nouns

                one_row=[df_aws_sentences.iloc[i,0],df_aws_sentences.iloc[i,1],df_aws_sentences.iloc[i,2],wordlow,translated,freq]
                df_word_freq.loc[windex_]=one_row
                windex_+=1
            #print(windex_)

    df_word_freq[['sentence_id','freq']]=df_word_freq[['sentence_id','freq']].astype(int)
    df_word_freq.rename(columns={'reviewsentence_wid':'textsentence_id','review_wid':'text_id'},inplace=True)
    #######################################################

    

    # give indices a name
    #df.index.rename('Index')
    
    ############################################################################
    ## output file generation and moving the Input file from Processed part
    ## this basically will be what the users need
    sentences_name_to_save=files_in_folder[ifiles][:-4]+'_ouput.csv'
    output_sentence_directory=output_directory+sentences_name_to_save
    df_aws_sentences.to_csv(output_sentence_directory,index=False)


    ###################################
    # merge 2 tables  -- in this part
    df_merged_output=df_aws_sentences.merge(df_word_freq,how='left',on=['textsentence_id','text_id','sentence_id'])
    merged_name_to_save=files_in_folder[ifiles][:-4]+'_output_words.csv'
    merged_output_directory=output_directory+merged_name_to_save
    df_merged_output.to_csv(merged_output_directory,index=False)

    # move processed file into folder named 
    shutil.move(file_directory,processed_directory)


    ## 2 output files are generated -- 
    # 1. sentences with sentiments
    # 2. merged table which contains word frequency-- this maybe used for visualisation in Tableau 


'''



# this is the code needed for running the sentiment analysis engine
import pandas as pd
import re
from langdetect import detect
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
#from textblob import TextBlob,
from textblob import Word, Blobber
import textblob_fr
import textblob_de
import os
import shutil # from removing files from one folder to another
os.environ["NLS_LANG"] = ".AL32UTF8" 
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from wordcloud import WordCloud
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
input_directory='/home/ayaqubov/engines/new_engine/DriverShare/Input/'
files_in_folder=os.listdir(input_directory)
#for i in range()
#files_in_folder

output_directory='/home/ayaqubov/engines/new_engine/DriverShare/Output/'
processed_directory='/home/ayaqubov/engines/new_engine/DriverShare/Processed/'


# check while the input is empty do the 

#while(os.listdir('/home/ayaqubov/engines/Windows-share/Input/')!=""):
# get the data and process the data

for ifiles in range(0,len(files_in_folder)):
    file_directory=input_directory + files_in_folder[ifiles]
    print(file_directory)
    #file_directory
    mycolumnsdf=['REVIEW_WID','REVIEW_FULL_TEXT']
    df_aws_reviews=pd.DataFrame(columns=mycolumnsdf)
    df_aws_reviews=pd.read_csv(file_directory)
    df_aws_reviews['LANGUAGE']=""
    num_unique_reviews=df_aws_reviews.shape[0]

    ###########################################################
    # here comes the check the language part and write correspondingly to the df_aws_reviews dataframe

    from nltk.tokenize import sent_tokenize, word_tokenize
    ## data processing
    print('Processing...')
    index=0
    badcounter=0
    for u in range(0,num_unique_reviews):
        review_id=df_aws_reviews.iloc[u,0]
        print(review_id)
        review_text=df_aws_reviews.iloc[u,1]
        review_str=str(review_text)
        #review_strw=review_str.encode('cp1252')## windows encoded
        #ureview_str=review_strw.decode('utf-8')
        #ureview_str=unicode(review_str,"utf-8")
        try:
            #ureview_str=review_str.decode('cp1252').encode('utf-8',errors='ignore')
            ureview_str=review_str.encode('utf-8',errors='ignore')
            ### windows does not support utf-8, keep in mind that instead it uses utf-16.
            #### Keep this in mind when using documents created on a Windows machine or saving files in a Windows machine.

            #print(review_str)
            #print(len(review_str))
            contain_l=re.search('[a-zA-Z]', review_str)
            if(contain_l!='None'):
                # handle japanese,arabic,chinese cases because they appear as the ?? marks in the results
                try:
                    text_lang=detect(ureview_str)
                    #text_lang2=identify_lang(review_str)
                except:
                    print 'Error in reading, keep reading'
                    #i_debug+=1
                    continue
                #check_words=word_tokenize(review_str)
                #for iiii in range(0,len(check_the_words)):
                #    if (check_the_words[iiii]=='product' or check_the_words[iiii]=='excellent'  or check_the_words[iiii]=='mouse'):
                #        break
                #            #continue

                if(text_lang=='en'):
                    df_aws_reviews.iloc[u,2]='English'
                    index+=1

                if(text_lang=='de'):
                    df_aws_reviews.iloc[u,2]='German'
                    index+=1

                if(text_lang=='fr'):
                    if('product' in ureview_str or 'excellent' in ureview_str or 'mouse' in ureview_str):
                        df_aws_reviews.iloc[u,2]='English'
                    else:
                        df_aws_reviews.iloc[u,2]='French'
                    index+=1
        except:
            # windows encoded file is handled seperately
            ureview_str=review_str.decode('cp1252').encode('utf-8',errors='ignore')
            #ureview_str=review_str.encode('utf-8',errors='ignore')
            ### windows does not support utf-8, keep in mind that instead it uses utf-16.
            #### Keep this in mind when using documents created on a Windows machine or saving files in a Windows machine.

            #print(review_str)
            #print(len(review_str))
            contain_l=re.search('[a-zA-Z]', review_str)
            if(contain_l!='None'):
                # handle japanese,arabic,chinese cases because they appear as the ?? marks in the results
                try:
                    text_lang=detect(ureview_str)
                    #text_lang2=identify_lang(review_str)
                except:
                    print 'Error in reading, keep reading'
                    #i_debug+=1
                    continue
                #check_words=word_tokenize(review_str)
                #for iiii in range(0,len(check_the_words)):
                #    if (check_the_words[iiii]=='product' or check_the_words[iiii]=='excellent'  or check_the_words[iiii]=='mouse'):
                #        break
                #            #continue

                if(text_lang=='en'):
                    df_aws_reviews.iloc[u,2]='English'
                    index+=1

                if(text_lang=='de'):
                    df_aws_reviews.iloc[u,2]='German'
                    index+=1

                if(text_lang=='fr'):
                    if('product' in ureview_str or 'excellent' in ureview_str or 'mouse' in ureview_str):
                        df_aws_reviews.iloc[u,2]='English'
                    else:
                        df_aws_reviews.iloc[u,2]='French'
                    index+=1
            print('Error happened in character decoding process::UnicodeDecodeError')
            badcounter+=1
        print('Counter bad unicodes are :: ', badcounter)

    # delete reviews that does not have language information
    # can also use the drop function from the pandas library
    df_aws_reviews=df_aws_reviews[df_aws_reviews['LANGUAGE'] != ""]
    df_aws_reviews.shape

    #functions to process the data:
    def count_words(sentence):
        words=word_tokenize(sentence)
        l=len(words)
        return l

    def get_sentiment_en(sentence):
        from textblob import TextBlob
        # this function is the simplest function
        blob=TextBlob(sentence)
        sentiment_pol=blob.sentiment.polarity
        #sentiment_sub=TextBlob(sentence).sentiment.subjectivity
        #sentiment_=sentiment_pol
        return sentiment_pol
    from textblob_fr import PatternTagger, PatternAnalyzer
    from textblob import Blobber
    def get_sentiment_fr(sentence):
        mysentence=str(sentence)
        tb = Blobber(pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        sentim = tb(mysentence).sentiment
        sentiment_pol=sentim[0]
        sentiment_sub=sentim[1]
        return sentiment_pol

    from textblob_de import TextBlobDE as TextBlob
    from textblob_de import PatternTagger
    def get_sentiment_de(sentence):
        blob = TextBlob(sentence)
        sentim=blob.sentiment
        sentiment_pol=sentim[0]
        sentiment_sub=sentim[1]
        return sentiment_pol
    import langid
    def identify_lang(sentence):
        cl=langid.classify(sentence)
        lan=cl[0]
        return lan
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    ps = PorterStemmer()
    def stem_word(in_word):
        stemmed_word=ps.stem(in_word)
        return stemmed_word




    #########################################################
    # division into the sentences
    df_num_of_rows=df_aws_reviews.shape[0]
    #mycolumns=['REVIEWSENTENCE_WID','REVIEW_WID','SENTENCE_ID','SKU','COUNTRY','SITE_URL','REVIEW_POSTED_DATE','WORD_COUNT',
    #                    'SENTIMENT','STAR_RATING','SENTENCE','PRODUCT_TYPE','PRODUCT_GROUP','PRODUCT_LINE_NAME']
    mycolumns_sentence=['textsentence_id','text_id','sentence_id','word_count',
                        'sentiment','sentence','language']
    df_aws_sentences=pd.DataFrame(columns=mycolumns_sentence)

    # adding sentences
    index_=0
    for i in range(0,df_num_of_rows):
        this_review=df_aws_reviews.iloc[i,1]
            # use try except because errpr was occuring in some cases
        try:
            sentences_this_review=sent_tokenize(this_review)
        except:
            print("Error in sentence tokenizing")
            continue
        num_of_sents=len(sentences_this_review)
        current_review_id=str(df_aws_reviews.iloc[i,0])
        #print(current_review_id)
        if(num_of_sents!=0):
            sent_id=0
            for j in range(0,num_of_sents):
                current_sentence=sentences_this_review[j]
                if(current_sentence in ["!","?","."]):
                    continue
                word_count=count_words(current_sentence)
                reviewsentence_id=current_review_id+'_'+str(sent_id)#int(current_review_id+'_'+str(sent_id))
                # Now calculate the polarity of sentece:
                #sentiment_=get_sentiment(current_sentence)
                if(df_aws_reviews.iloc[i,2]=='English'):
                    sentiment_=get_sentiment_en(current_sentence)
                elif(df_aws_reviews.iloc[i,2]=='French'):
                    sentiment_=get_sentiment_fr(current_sentence)
                elif(df_aws_reviews.iloc[i,2]=='German'):
                    sentiment_=get_sentiment_de(current_sentence)

                one_row=[reviewsentence_id,current_review_id,sent_id,word_count,sentiment_,current_sentence,df_aws_reviews.iloc[i,2]]
                df_aws_sentences.loc[index_]=one_row
                sent_id+=1
                index_+=1

    df_aws_sentences[['sentence_id','word_count']]=df_aws_sentences[['sentence_id','word_count']].astype(int)
    ########################################################
    ## now word frequency table

    cols_word_freq= ['reviewsentence_wid','review_wid','sentence_id','word','translated_word','freq']
    df_sents_num_of_rows=df_aws_sentences.shape[0]


    df_word_freq=pd.DataFrame(columns=cols_word_freq)
    # get rid of commas etc
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    import re
    def check_num(input_s): 
        num_format = re.compile("^[\-]?[1-9][0-9]*\.?[0-9]+$")
        isnumber = re.match(num_format,input_s)
        #isnumber=~
        if isnumber:
            return True
        else:
            return False
    def check_letter(input_s):
        #remove len 1 and 2s(come back for len 2 later)
        l=len(input_s)
        if(l==1 or l==2):
            return True
        return False

    from nltk.corpus import stopwords
    #stopwords_ = set(stopwords.words('english'))
    stopwords_fr=set(stopwords.words('french'))
    stopwords_en=set(stopwords.words('english'))
    stopwords_ge=set(stopwords.words('german'))


    windex_=0

    for i in range(0,df_sents_num_of_rows):
        #print(i)
        sentence=df_aws_sentences.iloc[i,5]
        #words=word_tokenize(sentence)
        # maybe use try-except block as follows:
        #try:
        #words=tokenizer.tokenize(sentence)
        #except:
        #print("error in word tokenizing")
        words=tokenizer.tokenize(sentence)
        tags_=nltk.pos_tag(words)
        num_words=len(words)
        for j in range(0,num_words):
            word=words[j]
            wordlow=word.lower()
            # check if it is noun here
            translated=wordlow  ## for another language we need translation
            freq=1
            w_isnum=check_num(wordlow)
            one_two_let=check_letter(wordlow)
            if(df_aws_sentences.iloc[i,6]=='English'):
                stopwords_=stopwords_en
            if(df_aws_sentences.iloc[i,6]=='French'):
                stopwords_=stopwords_fr
            if(df_aws_sentences.iloc[i,6]=='German'):
                stopwords_=stopwords_ge
            if(wordlow in stopwords_ or w_isnum or one_two_let):
                continue


            if(tags_[j][1]=='NN' or tags_[j][1]=='NNS' or tags_[j][1]=='NNP'):
                # since we expect aspects are more likely to be among the nouns

                one_row=[df_aws_sentences.iloc[i,0],df_aws_sentences.iloc[i,1],df_aws_sentences.iloc[i,2],wordlow,translated,freq]
                df_word_freq.loc[windex_]=one_row
                windex_+=1
            #print(windex_)

    df_word_freq[['sentence_id','freq']]=df_word_freq[['sentence_id','freq']].astype(int)
    df_word_freq.rename(columns={'reviewsentence_wid':'textsentence_id','review_wid':'text_id'},inplace=True)
    #######################################################

    
    ####################################################################
    # make wordcloud out of word frequency table and save it
    mytext=list(df_word_freq['word'])
    smytext=" ".join(mytext)
    wordcloud1=WordCloud(background_color="white", max_words=1000).generate(smytext)
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud_name_to_save=output_directory+files_in_folder[ifiles][:-4]+'_wordcloud'
    plt.savefig(wordcloud_name_to_save)

    # give indices a name
    #df.index.rename('Index')
    
    ############################################################################
    ## output file generation and moving the Input file from Processed part
    ## this basically will be what the users need
    sentences_name_to_save=files_in_folder[ifiles][:-4]+'_ouput.csv'
    output_sentence_directory=output_directory+sentences_name_to_save
    df_aws_sentences.to_csv(output_sentence_directory,index=False)


    ###################################
    # merge 2 tables  -- in this part
    df_merged_output=df_aws_sentences.merge(df_word_freq,how='left',on=['textsentence_id','text_id','sentence_id'])
    merged_name_to_save=files_in_folder[ifiles][:-4]+'_output_words.csv'
    merged_output_directory=output_directory+merged_name_to_save
    df_merged_output.to_csv(merged_output_directory,index=False)

    # move processed file into folder named 
    shutil.move(file_directory,processed_directory)


    ## 2 output files are generated -- 
    # 1. sentences with sentiments
    # 2. merged table which contains word frequency-- this maybe used for visualisation in Tableau 
