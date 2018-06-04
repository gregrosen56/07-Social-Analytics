
Observable Trends

1)  I was surprised to see, especially given the sample images for the exercise, that none of the five news organisations crossed above the 0 threshold in its mean compound score.  

2)  Six of the ten most negative tweets retrieved contained the word 'murder.'  The remaining tweets of those ten contained the words 'war', 'rape', 'dead', and 'terrorist'.  The most positive tweets contain words like 'free', 'God', 'star', and 'sweet'.

3)  I investigated the large number of tweets with a polarity score of zero (155 in my data set).  At first glance, most of them do not seem to exhibit anything that makes them stand out from the other tweets that VADER was able to properly analyze, and many of them certainly contain words that would move them away from a perfect 1 score for NEUTRAL.  I have to assume VADER is not functioning properly all of the time.   


```python
# DEPENDENCIES
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.axes as ax
import datetime

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# CYCLE THROUGH 100 TWEETS FROM EACH OF FIVE NEWS ORGANIZATIONS

NewsOrgs = ("BBCBreaking", 
            "CBSNews", 
            "CNN",
            "FoxNews", 
            "nytimes")

sentiment_list = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
text_list = []
date_list = []
counter_list = []
org_list = []

for org in NewsOrgs:
    
    counter = 0
        
    for x in range(1, 6):
        
        public_tweets = api.user_timeline(org, page=x)
        
        for tweet in public_tweets:
    
            print(str(org) + " " + str(counter))
            print(tweet['text'])
            print('------------------------------')
            
            # Analyze sentiment of tweet
            results = analyzer.polarity_scores(tweet['text'])
            compound = results['compound']
            pos = results['pos']
            neu = results['neu']
            neg = results['neg']
            text = tweet['text']
            date = tweet['created_at']
            ago = counter
            
            # Add each value to the appropriate list
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            text_list.append(text)
            date_list.append(date)
            counter_list.append(ago)
            org_list.append(org)
            
            counter = counter + 1
        
    sentiment = {"User":org,
                     "Compound":round(np.mean(compound_list),3),
                     "Positive":round(np.mean(positive_list),3),
                     "Negative":round(np.mean(negative_list),3),
                     "Neutral":round(np.mean(neutral_list),3)} 
    sentiment_list.append(sentiment)  
```

    BBCBreaking 0
    A minute's silence falls across the UK as the victims of the #LondonBridge terror attack are remembered, one year o… https://t.co/jJPzEmXsNq
    ------------------------------
    BBCBreaking 1
    Armed police cordon off Berlin Cathedral after an officer reportedly shot a man at the building https://t.co/DdbPH9rZVJ
    ------------------------------
    BBCBreaking 2
    Visa says cardholders should now be able to use their cards, with systems operating at "close to normal" levels, bl… https://t.co/tUahgaGGr0
    ------------------------------
    BBCBreaking 3
    Donald Trump's summit with Kim Jong-un on 12 June is back on, the US president says, a week after it was scrapped https://t.co/WfevC3zR8i
    ------------------------------
    BBCBreaking 4
    Visa says some card payments are currently failing across the UK and elsewhere in Europe https://t.co/3IJ9nksdEc
    ------------------------------
    BBCBreaking 5
    Five big cats which escaped from a zoo in western Germany have been recaptured, reportedly with the help of a drone https://t.co/g32tKvh2qT
    ------------------------------
    BBCBreaking 6
    Two lions, two tigers and a jaguar escape from a German zoo, with public warned to stay indoors, police say https://t.co/qUowA6ZZ4a
    ------------------------------
    BBCBreaking 7
    Spain's prime minister is forced out of his job, losing a no-confidence vote triggered by a corruption scandal https://t.co/T6sskEuPSj
    ------------------------------
    BBCBreaking 8
    Talks on summit between Trump and Kim Jong-un "moving in the right direction", secretary of state Mike Pompeo says https://t.co/PnxigeVi40
    ------------------------------
    BBCBreaking 9
    The US to put tariffs on steel and aluminium made by key allies, a move France says is "unjustifiable and dangerous" https://t.co/RuuMpmGgNJ
    ------------------------------
    BBCBreaking 10
    RT @BBCSport: Zinedine Zidane says he is stepping down as Real Madrid boss, just five days after leading them to a third straight Champions…
    ------------------------------
    BBCBreaking 11
    Lithuania and Romania assisted CIA torture of al-Qaeda suspects, European Court of Human Rights rules https://t.co/VdnWsfigYS
    ------------------------------
    BBCBreaking 12
    The moment Russian journalist Arkady Babchenko, who had been reported dead yesterday, appeared at a press conferenc… https://t.co/iHb6Rc1NCY
    ------------------------------
    BBCBreaking 13
    "Assassination" of journalist Arkady Babchenko was "staged to expose Russian agents", says head of Ukraine's securi… https://t.co/SPcn5LgkRh
    ------------------------------
    BBCBreaking 14
    Russian journalist Arkady Babchenko, reportedly assassinated in Kiev on Tuesday, appears on TV alive and well https://t.co/Jq3liX4P2o
    ------------------------------
    BBCBreaking 15
    "Abhorrent, repugnant" - and cancelled. ABC drops Roseanne's show after star's racist tweet about an Obama aide https://t.co/2eutMyKysP
    ------------------------------
    BBCBreaking 16
    Hurricane Maria killed 4,600 people in Puerto Rico, 70 times the official toll - Harvard study https://t.co/3HplA9VkAB
    ------------------------------
    BBCBreaking 17
    Two police officers killed by gunman in Liège were both women, city's mayor confirms
    https://t.co/WyTQaq20ss
    ------------------------------
    BBCBreaking 18
    Belgian authorities confirm they are treating the attack in Liège as terrorism 
    https://t.co/d4uJcconBF
    ------------------------------
    BBCBreaking 19
    Two police officers shot dead in eastern Belgian city of Liège, senate chief says, with reports attacker is killed https://t.co/FTmHJQ5V71
    ------------------------------
    BBCBreaking 20
    Murder investigation launched after man was killed and several more injured when car hit people near Stockport club https://t.co/raUxSrvGY7
    ------------------------------
    BBCBreaking 21
    Man in his 80s dies after vehicle submerged in flood water in Walsall, police say https://t.co/8hmZZR6Ifh
    ------------------------------
    BBCBreaking 22
    Man arrested on suspicion of murder of woman, 31, and girl, 11, in Gloucester https://t.co/5W9CiZIqiz
    ------------------------------
    BBCBreaking 23
    After his dramatic rescue of a young boy from a Paris balcony, Malian migrant Mamoudou Gassama is made a French cit… https://t.co/ki7aWgGtPQ
    ------------------------------
    BBCBreaking 24
    Italy PM-designate Guiseppe Conte gives up bid to form government amid reports president vetoed economy pick https://t.co/AbEoT2SdF0
    ------------------------------
    BBCBreaking 25
    RT @BBCSport: 🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆🏆
    
    It's heartbreak for Liverpool as Real Madrid win their 13th Champions League title.
    
    FT 3-1
    
    https://t.co/Hr5J…
    ------------------------------
    BBCBreaking 26
    The final result in Ireland's abortion reform referendum
    
    Yes 1,429,981
    No 723,632
    
    https://t.co/urPAqRybLz https://t.co/8665LHVeOC
    ------------------------------
    BBCBreaking 27
    Official result: Republic of Ireland votes resoundingly to overturn abortion ban with voters backing reform of cons… https://t.co/1ghSuV99Ff
    ------------------------------
    BBCBreaking 28
    Irish PM hails "quiet revolution" as early results point to "resounding" referendum vote for overturning abortion b… https://t.co/fW4uTOgwOc
    ------------------------------
    BBCBreaking 29
    Leaders of both Koreas meet in border zone as efforts continue to reschedule Donald Trump's summit with Kim Jong-un https://t.co/ZNdh7ZZZAk
    ------------------------------
    BBCBreaking 30
    'No' campaign spokesman accepts defeat in Ireland referendum which will allow liberalisation of abortion law… https://t.co/b8Fn4jPkTj
    ------------------------------
    BBCBreaking 31
    Exit polls suggest the Republic of Ireland has voted by a landslide to liberalise its strict abortion laws https://t.co/FeHK5CKrvI
    ------------------------------
    BBCBreaking 32
    A 95-year-old man arrested on suspicion of murder after female carer died of head injuries, London police say https://t.co/Wttf0ezBRU
    ------------------------------
    BBCBreaking 33
    Harvey Weinstein is charged with rape and a criminal sex act on two women, after months of sexual abuse allegations https://t.co/xKlFuMthOA
    ------------------------------
    BBCBreaking 34
    Film producer Harvey Weinstein arrives at NY police station where he's expected to answer sexual misconduct charges https://t.co/SVJvYQ0BCT
    ------------------------------
    BBCBreaking 35
    Man who attacked D-Day veteran, 96, with hammer during raid on his home in Somerset found guilty of attempted murder https://t.co/LXYx1mLb9y
    ------------------------------
    BBCBreaking 36
    Australia and Netherlands formally accuse Russia of responsibility for downing MH17 passenger jet in 2014 https://t.co/zyMC4Ccbyz
    ------------------------------
    BBCBreaking 37
    Hollywood mogul Harvey Weinstein expected to surrender to police on sexual misconduct charges, US media report https://t.co/NzeaFHi9FZ
    ------------------------------
    BBCBreaking 38
    "Hopefully positive things will be taking place... but if they don't, we are more ready than we have ever been befo… https://t.co/jgzyXyqNwm
    ------------------------------
    BBCBreaking 39
    Two men jailed for life for murder of four children in petrol bomb attack in Greater Manchester https://t.co/fEgISmD1y5
    ------------------------------
    BBCBreaking 40
    "You talk about your nuclear capabilities, but ours are so massive and powerful that I pray to God they will never… https://t.co/NmbqVRuyAV
    ------------------------------
    BBCBreaking 41
    In letter to Kim Jong-un, President Trump says "tremendous anger and open hostility" displayed in North Korea state… https://t.co/4P6aLp1LXE
    ------------------------------
    BBCBreaking 42
    US-North Korea summit will not take place, Donald Trump says https://t.co/Kzej5c101k
    ------------------------------
    BBCBreaking 43
    Army sergeant Emile Cilliers guilty of trying to murder his wife by tampering with her parachute https://t.co/OdTAyBSJZo
    ------------------------------
    BBCBreaking 44
    North Korea appears to have blown up tunnels at its only nuclear test site, a key move to reducing regional tensions https://t.co/s92Pw9KS4O
    ------------------------------
    BBCBreaking 45
    A couple have been found guilty of murdering their French au pair and burning her body in their garden https://t.co/8r6yRdCG4B
    ------------------------------
    BBCBreaking 46
    A missile fired at flight MH17 in 2014, killing 298 people, belonged to a Russian brigade, investigators say https://t.co/yLua7Ztfui
    ------------------------------
    BBCBreaking 47
    A couple have been found guilty of murdering their French au pair and burning her body in their London garden
    
    https://t.co/ZRwyexYiwh
    ------------------------------
    BBCBreaking 48
    A missile fired at flight MH17 in 2014, killing 298 people, belonged to a Russian brigade, investigators say https://t.co/D8VQAg3Ca1
    ------------------------------
    BBCBreaking 49
    Man arrested in Hertfordshire on suspicion of preparing terrorist acts, police say https://t.co/SCbyRXldte
    ------------------------------
    BBCBreaking 50
    Yulia Skripal, poisoned daughter of Russian ex-spy Sergei, says she is lucky to be alive after attack https://t.co/Pe34d0cCOr
    ------------------------------
    BBCBreaking 51
    After the alleged payment to Cohen for White House talks, Ukraine’s Anti Corruption Bureau drops Paul Manafort inve… https://t.co/uZrdnv8IBK
    ------------------------------
    BBCBreaking 52
    Cohen received at least $400,000 to fix talks between Ukraine’s Poroshenko and President Trump – intelligence sourc… https://t.co/WAWUjJQmHP
    ------------------------------
    BBCBreaking 53
    BBC Exclusive: Trump lawyer Michael Cohen “paid by Ukraine” to arrange White House talks in June 2017
    
    https://t.co/NsfSjRzEzJ
    ------------------------------
    BBCBreaking 54
    Berlinah Wallace given life sentence with minimum term of 12 years for throwing sulphuric acid over former partner,… https://t.co/hu686gzzht
    ------------------------------
    BBCBreaking 55
    RT @BBCSport: 🔴 Confirmed 🔴
    
    Arsenal have announced Unai Emery as Arsene Wenger's successor.
    
    More here👉 https://t.co/x1kOq2Nos3 https://t.…
    ------------------------------
    BBCBreaking 56
    "There has been a change in customer shopping habits" - Marks and Spencer Chief Executive Steve Rowe on 62% fall in… https://t.co/3bTbpSQHid
    ------------------------------
    BBCBreaking 57
    "The great American novelist of our postwar world" - Tributes paid to novelist Philip Roth, who has died aged 85 https://t.co/vaxhh5ZPCe
    ------------------------------
    BBCBreaking 58
    President Trump says his historic summit with North Korea's Kim Jong-un next month may be delayed https://t.co/bzBWdJINR0
    ------------------------------
    BBCBreaking 59
    Supermarket giant Tesco plans to close its Tesco Direct website, putting 500 jobs at risk https://t.co/QiEbKrn4aG
    ------------------------------
    BBCBreaking 60
    UK holds a minute's silence to remember the victims of the #ManchesterArena attack one year on… https://t.co/pV1AULBovR
    ------------------------------
    BBCBreaking 61
    Marks and Spencer to close 100 stores by 2022 saying reorganisation "vital" for firm's future https://t.co/2LBhVTOMgD
    ------------------------------
    BBCBreaking 62
    RT @BBCSport: Arsenal are set to appoint Unai Emery as their new manager.
    
    Read more
    👉 https://t.co/Z4g4asmx3i #Arsenal https://t.co/hxcv73…
    ------------------------------
    BBCBreaking 63
    Ken Livingstone says he is resigning from the Labour party amid anti-Semitism row
    https://t.co/Pv17B4rd7I
    ------------------------------
    BBCBreaking 64
    Duke and Duchess of Sussex release official photographs from their wedding day
    
    https://t.co/6yvDshkVs8… https://t.co/SEEvwJlnXr
    ------------------------------
    BBCBreaking 65
    A 72-second silence in memory of the victims marks the beginning of inquiry into Grenfell Tower fire https://t.co/yw2uIVMiyh
    ------------------------------
    BBCBreaking 66
    Jailed British-Iranian Nazanin Zaghari-Ratcliffe charged in Iran with spreading propaganda, her campaign says https://t.co/oOU3I7SkC8
    ------------------------------
    BBCBreaking 67
    One of black boxes from Boeing 737 that crashed in Havana on Friday found "in good condition", Cuban officials say. https://t.co/OHYW3Dsy6l
    ------------------------------
    BBCBreaking 68
    RT @BBCNews: Harry and his new wife Meghan have left Windsor Castle in an open-top classic sports car for their evening wedding reception a…
    ------------------------------
    BBCBreaking 69
    RT @BBCSport: CHELSEA HAVE WON THE FA CUP! 🏆
    
    They've beaten Man Utd 1-0.
    
    Party time for the Blues 🎉
    
    👉 https://t.co/zzvVrq4qSz #FACupFina…
    ------------------------------
    BBCBreaking 70
    RT @BBCSport: Full-Time: Celtic 2-0 Motherwell. 
    
    Celtic complete back-to-back trebles by clinching the Scottish Cup. Goals from Callum McG…
    ------------------------------
    BBCBreaking 71
    Enormous crowds greet the Duke &amp; Duchess of Sussex as #harryandmeghan head up Long Walk and back to Windsor Castle… https://t.co/pIohUoEeqa
    ------------------------------
    BBCBreaking 72
    The royal couple leave St George's Chapel - and here's the kiss!
    
    https://t.co/XbDbyZPqJN #royalwedding https://t.co/wpcWWl3aDL
    ------------------------------
    BBCBreaking 73
    The royal couple have made their vows and exchanged rings.
    
    Harry &amp; Meghan are married! 🎉
    
    https://t.co/XbDbyZPqJN… https://t.co/n9C0CnBYRV
    ------------------------------
    BBCBreaking 74
    And the bride arrives - Meghan Markle emerges into bright May sunshine at St George's Chapel… https://t.co/TJCRTwy4KG
    ------------------------------
    BBCBreaking 75
    A stunning sight! Meghan Markle's car drives down Long Walk to Windsor Castle ahead of the #royalwedding… https://t.co/L2YsIw0oFY
    ------------------------------
    BBCBreaking 76
    Prince Harry arrives at Windsor Castle with his brother and best man, Prince William
    
    https://t.co/XbDbz071Bl… https://t.co/i6YdflPlVE
    ------------------------------
    BBCBreaking 77
    The first glimpse of bride Meghan Markle as she leaves Cliveden House Hotel for Windsor Castle… https://t.co/Itztp6zP8I
    ------------------------------
    BBCBreaking 78
    Prince Harry &amp; Meghan Markle will become the Duke &amp; Duchess of Sussex on #royalwedding day, Buckingham Palace annou… https://t.co/3Jzux7S7Tl
    ------------------------------
    BBCBreaking 79
    10 people have died, and another 10 were injured in shooting at Santa Fe High School - Texas Governor Greg Abbott s… https://t.co/rILeNeQhke
    ------------------------------
    BBCBreaking 80
    More than 100 people feared dead after a Boeing 737 airliner crashed and exploded near Havana airport, Cuban state… https://t.co/uk73F47FkS
    ------------------------------
    BBCBreaking 81
    Boeing 737 crashes shortly after take-off from Havana, Cuban media report https://t.co/VwVZvReA7h
    ------------------------------
    BBCBreaking 82
    "I just ran as fast as I could" - Santa Fe High School students describe shooting incident in Texas that has left u… https://t.co/Ek2vfRZaMo
    ------------------------------
    BBCBreaking 83
    Santa Fe High School shooting update:
    
    - Between eight and 10 people killed in incident, say police
    - Majority of d… https://t.co/655gYTdsos
    ------------------------------
    BBCBreaking 84
    "This has been going on too long in our country, too many years, too many decades now" - President Trump on reports… https://t.co/L27O1tVHmU
    ------------------------------
    BBCBreaking 85
    "Multiple people have died" - @BBCJamesCook gives the latest on the Santa Fe High School shooting… https://t.co/TxEKQetc7s
    ------------------------------
    BBCBreaking 86
    At least eight people are dead after a shooting at Santa Fe High School in Texas, reports say https://t.co/qwC5MN8BLM
    ------------------------------
    BBCBreaking 87
    Russian ex-spy Sergei Skripal has been discharged from hospital after being poisoned with a nerve agent in Salisbury https://t.co/m09vK9vPg5
    ------------------------------
    BBCBreaking 88
    The Prince of Wales will accompany Meghan Markle down the aisle, Kensington Palace confirms https://t.co/lF1tHcOJ2I #royalwedding
    ------------------------------
    BBCBreaking 89
    Explosive eruption at Hawaii’s Kilauea volcano sends ash thousands of metres into the sky https://t.co/DunkpP2Xpu
    ------------------------------
    BBCBreaking 90
    Kasim Lewis, 31, jailed for life with a minimum of 29 years at Old Bailey for murdering 22-year-old barmaid Iuliana… https://t.co/CL9DGlheZt
    ------------------------------
    BBCBreaking 91
    Berlinah Wallace found guilty of throwing corrosive substance with intent at Bristol Crown Court after injuring for… https://t.co/jXEDp1FCkj
    ------------------------------
    BBCBreaking 92
    Berlinah Wallace, 48, cleared of killing boyfriend in acid attack in Bristol in 2015 https://t.co/mG58fZAScs
    ------------------------------
    BBCBreaking 93
    UK government to consult on banning flammable cladding, after Grenfell review stopped short of recommending ban https://t.co/MEckLQ9heM
    ------------------------------
    BBCBreaking 94
    Meghan Markle says she hopes her father "can be given the space he needs to focus on his health"… https://t.co/J1fkcc7X1M
    ------------------------------
    BBCBreaking 95
    Meghan Markle says her father will not now be attending her wedding to Prince Harry on Saturday https://t.co/g5w2J6kr2V
    ------------------------------
    BBCBreaking 96
    Grenfell Tower fire review concludes indifference and ignorance led "race to the bottom" in building safety practic… https://t.co/haedS7P28u
    ------------------------------
    BBCBreaking 97
    UK retailer Mothercare says it is to shut 50 stores as part of a restructuring plan, resulting in
    hundreds of job l… https://t.co/mRqaZEXXH8
    ------------------------------
    BBCBreaking 98
    The maximum bet on fixed-odds betting terminals will be reduced from £100 to £2, the government says https://t.co/wLmS7JM1EO
    ------------------------------
    BBCBreaking 99
    Oxfam chief executive Mark Goldring to stand down after scandal over claims of sexual misconduct by staff in Haiti https://t.co/8xQExjxF8t
    ------------------------------
    CBSNews 0
    Delta Air Lines is "conducting a thorough review of the situation" after a Pomeranian on a flight from Newark to Ph… https://t.co/yptBqi6CPL
    ------------------------------
    CBSNews 1
    Jury acquits deputy, awards $4 to family in wrongful death suit of 30-year-old father https://t.co/Ox6TNQdmbg https://t.co/79JaEc6iKC
    ------------------------------
    CBSNews 2
    Indiana police officers gather for the high school graduation of the daughter of a fallen trooper… https://t.co/NCJl207aYg
    ------------------------------
    CBSNews 3
    Brush fire forces thousands to evacuate parts of Southern California https://t.co/4LzRj0Rf6e
    ------------------------------
    CBSNews 4
    "It's not one person and then it stops": Lara Logan breaks silence on her sexual assault during the Egyptian Revolu… https://t.co/ZiC7I9xjJv
    ------------------------------
    CBSNews 5
    Fourth-grader's rendition of John Lennon's "Imagine" goes viral https://t.co/c26K8vFaee https://t.co/3xvCNtxJGq
    ------------------------------
    CBSNews 6
    Music of the streets: New York City's public pianos https://t.co/y3yK0f4CK0 https://t.co/UtTaa3Gyxx
    ------------------------------
    CBSNews 7
    Woman who went into labor during final exam graduates from Harvard Law School with her daughter in her arms… https://t.co/Pku4PaN5I0
    ------------------------------
    CBSNews 8
    Remembering 1968: Robert F. Kennedy, and a generation's loss https://t.co/Iqkap0ukfn https://t.co/6pV5NHsY6I
    ------------------------------
    CBSNews 9
    "I'm an old dog, and this is a new trick": Bill Clinton and James Patterson have co-authored a political thriller.… https://t.co/zp5ykvgxwJ
    ------------------------------
    CBSNews 10
    Roadside America: A tiny slice of Americana https://t.co/fLkaPdSaXu https://t.co/8HnBu63nqY
    ------------------------------
    CBSNews 11
    Not your average graduating senior, WWII veteran Roland Martineau missed getting his diploma in 1942, and finally g… https://t.co/RqdaZwOVmQ
    ------------------------------
    CBSNews 12
    Jim Gaffigan on why he doesn't care about the Triple Crown https://t.co/Y94n9ZOhpA https://t.co/ChWa1kxAsn
    ------------------------------
    CBSNews 13
    Bill Clinton and James Patterson co-author a political beach read https://t.co/mnHzkH507L https://t.co/ceFhuDkqCP
    ------------------------------
    CBSNews 14
    High school senior Tyler Solomon was moved to tears when his dad, who has been deployed overseas, surprised him at… https://t.co/ieSaL2Pk3c
    ------------------------------
    CBSNews 15
    The war in Afghanistan is the longest in US history. After over 16 years, a trillion dollars and 2400 American live… https://t.co/AJEhgD5IVI
    ------------------------------
    CBSNews 16
    Tonight on @60Minutes, meet Milwaukee Bucks star Giannis Antetokounmpo. Steve Kroft reports on the 23-year-old whos… https://t.co/n1yKVJV93K
    ------------------------------
    CBSNews 17
    2 climbers died after falling from Yosemite National Park's El Capitan, which looms 3,000 feet above the floor of Y… https://t.co/09wBAIkXe0
    ------------------------------
    CBSNews 18
    Tonight, celebrity chef José Andrés has prepared more hot meals in Puerto Rico after Hurricane Maria than any of th… https://t.co/Ocu6CR9EH0
    ------------------------------
    CBSNews 19
    Watch NBA Finals 2018: Cleveland Cavaliers vs. Golden State Warriors for the fourth consecutive year… https://t.co/8rmeHJVFAv
    ------------------------------
    CBSNews 20
    Jimmy Fallon surprises Parkland students at high school graduation: "Don't let anything stop you"… https://t.co/MbLpAlhT9N
    ------------------------------
    CBSNews 21
    Woman held gun to head, fired shots near San Diego marathon, police say https://t.co/zp2Xosaaeg https://t.co/SPJ1cyimW4
    ------------------------------
    CBSNews 22
    Woman admitted to shooting and killing husband, says it was because he beat family cat, police say… https://t.co/ZZPmyGZFSB
    ------------------------------
    CBSNews 23
    Hang glider pilot dies in Idaho air show accident https://t.co/AlXgBl0PUQ https://t.co/bvrfnvMZ0V
    ------------------------------
    CBSNews 24
    Remembering 1968: The loss of RFK fifty years ago https://t.co/3cZJURGRZN https://t.co/2g5mlSi8mD
    ------------------------------
    CBSNews 25
    Woman who went into labor during final exam graduates from Harvard Law School with her daughter in her arms… https://t.co/8dSfQgQbZO
    ------------------------------
    CBSNews 26
    "You may say I'm a dreamer..."
    This 10-year-old performed John Lennon's "Imagine" for his school talent show and a… https://t.co/Qrhpn2XI8w
    ------------------------------
    CBSNews 27
    Stephen Stills and Judy Collins playing beautiful music together after breakup years ago https://t.co/huVOY3TTHp https://t.co/0vrjsmVBgr
    ------------------------------
    CBSNews 28
    One third of the Senate is up for election this year.
    Here are seven Senate midterm elections you should watch out… https://t.co/gL1txKxySU
    ------------------------------
    CBSNews 29
    Former President Bill Clinton and the world's bestselling author James Patterson co-author a political thriller… https://t.co/cSEtXfx79P
    ------------------------------
    CBSNews 30
    FBI agent under investigation for accidental shooting at Denver nightclub https://t.co/dSpxmCiEC5 https://t.co/RN40AF71Sj
    ------------------------------
    CBSNews 31
    Woman faces manslaughter charges after deadly hit-and-run at little league baseball field in Maine… https://t.co/ELwPLNNfiI
    ------------------------------
    CBSNews 32
    Rep. Will Hurd says he thinks it would be a "terrible move" for President Trump to pardon himself. 
    
    "I think peopl… https://t.co/Z6UkRGwWgZ
    ------------------------------
    CBSNews 33
    Who will pay for Kim Jong Un's hotel stay in Singapore during summit with President Trump? https://t.co/vwma70d4qg https://t.co/HgbBOdCAr3
    ------------------------------
    CBSNews 34
    Fourth-grader's rendition of John Lennon's "Imagine" at school talent show goes viral https://t.co/B3D7QUCvtO https://t.co/kvzqTR9a1n
    ------------------------------
    CBSNews 35
    North Korea experts' advice to President Trump: Keep summit as simple as possible https://t.co/Acudj31YOf https://t.co/UdtLZYezi1
    ------------------------------
    CBSNews 36
    Rudy Giuliani adamant that President Trump would fight Mueller subpoena to testify in Russia probe… https://t.co/KD7y72xKYa
    ------------------------------
    CBSNews 37
    More than six in 10 women voters in battleground districts say the president does not respect or understand people… https://t.co/sVwz3d4Le1
    ------------------------------
    CBSNews 38
    Rep. Will Hurd says GOP members have the votes to bring immigration bill to the House floor.
    
    "Now is time to solve… https://t.co/q4wgmiJwnd
    ------------------------------
    CBSNews 39
    Single mom who went into labor during final exam graduates from Harvard Law School https://t.co/GgBj9FyMM3 https://t.co/tSt7ZXjSJ6
    ------------------------------
    CBSNews 40
    Ohio Gov. John Kasich says he is "very, very concerned" about North Korea summit, and the U.S. must be "extremely c… https://t.co/en3DRP869u
    ------------------------------
    CBSNews 41
    Kasich scolds GOP leaders for thinking "they have to ask permission from the president to do anything." (via… https://t.co/RhcK8pjljf
    ------------------------------
    CBSNews 42
    "I have been frankly shocked at the fact that our leaders think they have to ask permission from the president to d… https://t.co/n9OZmxtIga
    ------------------------------
    CBSNews 43
    CBS News Battleground Tracker: A closer look at the parties 
    
    In battleground districts where control of House will… https://t.co/Fv6yBCiwRU
    ------------------------------
    CBSNews 44
    2018 CBS News Battleground Tracker: Big gender gap 
    
    There is a big gender gap in competitive districts: Democrats… https://t.co/lFkYQFjNQg
    ------------------------------
    CBSNews 45
    CBS News Battleground Tracker: The House is a toss-up 
    
    CBS News' estimate is Democrats 219 - Republicans 216 in ti… https://t.co/kIM8GDgNir
    ------------------------------
    CBSNews 46
    Idaho teacher charged after allegedly feeding puppy to snapping turtle in front of students https://t.co/C8IPr0GleL https://t.co/cUF6gykk40
    ------------------------------
    CBSNews 47
    Small plane with 4 on board crashes off New York's Long Island, FAA says https://t.co/uBz5vcVBZd https://t.co/fw4rCfh2c2
    ------------------------------
    CBSNews 48
    Many breast cancer patients can safely skip chemotherapy, major study finds https://t.co/Kcd2GeZe3j https://t.co/8gniV1gbQa
    ------------------------------
    CBSNews 49
    Dog found dead inside cargo hold on Delta Airlines flight. He was an 8-year-old Pomeranian named Alejandro… https://t.co/dBdq8En9Sq
    ------------------------------
    CBSNews 50
    Fire erupts at scrap yard in Oakland, California, that's had trouble in the past https://t.co/FgeIs2Snuy https://t.co/6LQShqw9vq
    ------------------------------
    CBSNews 51
    Report: 14-year-old stabbed, killed after fight that started over sleepover https://t.co/Fzh8NupmWE https://t.co/4LMNSkBtsY
    ------------------------------
    CBSNews 52
    Far-right party leader calls Nazi era a "speck of bird poop" in German history https://t.co/J4eOVWWiPN https://t.co/fTlwpj5rER
    ------------------------------
    CBSNews 53
    U.S. citizen killed in Nicaragua amid violence and social unrest https://t.co/qpztwh5AKe https://t.co/sJZkd1qxDt
    ------------------------------
    CBSNews 54
    DNA from tissue in trash led to arrest in Golden State Killer case, records show https://t.co/bRiR2LYI7x https://t.co/ULMHh1Sv9k
    ------------------------------
    CBSNews 55
    Ohio Governor John Kasich scolds GOP leaders for thinking “they have to ask permission from the president to do any… https://t.co/gxEcambRce
    ------------------------------
    CBSNews 56
    Small plane with 4 on board crashes off New York's Long Island, FAA says https://t.co/oktGWB9b8q https://t.co/08flsJYLnZ
    ------------------------------
    CBSNews 57
    FBI agent's firearm accidentally discharges at Denver nightclub: police https://t.co/h2Oos8F3kf https://t.co/sOgO2Eglbd
    ------------------------------
    CBSNews 58
    Google result for Republican shows picture with "BIGOT" on it https://t.co/b9nCTgi1fL https://t.co/6CeQqRIOIR
    ------------------------------
    CBSNews 59
    President Trump hints at longer path for North Korea to denuke https://t.co/rhd5rn8ffW https://t.co/NYDo9tvlp4
    ------------------------------
    CBSNews 60
    Why are deaths among U.S. kids, teens on the rise? https://t.co/it8apo5Lc0 https://t.co/LiTb9Cpxh3
    ------------------------------
    CBSNews 61
    Shooting death of Steven Pitt, forensic psychiatrist, linked to killings of two women https://t.co/XIPKIsWKIU https://t.co/DkZmyowwLc
    ------------------------------
    CBSNews 62
    Cosby accuser Andrea Constand: "I did it for justice" https://t.co/t4Iqez7XYH https://t.co/yWDsM2zyMF
    ------------------------------
    CBSNews 63
    Hang glider killed in Idaho air show accident https://t.co/enpQJwB6dy https://t.co/EYUMtKdV5S
    ------------------------------
    CBSNews 64
    Four killed, seven injured in Oregon highway crash, police say https://t.co/ijsddWiVkP https://t.co/4jfZBVWt3h
    ------------------------------
    CBSNews 65
    Woman faces manslaughter charges after deadly hit-and-run at baseball game in Maine https://t.co/oAv5Kn3F9x https://t.co/AiPuRoW1TE
    ------------------------------
    CBSNews 66
    Roger Stone's former "social media strategist" testifies before grand jury https://t.co/4vnNieddUh https://t.co/GFth4De3Ot
    ------------------------------
    CBSNews 67
    Search continues for 2 teens believed to have drowned in Georgia https://t.co/ZcCQfnKrYS https://t.co/8Khgv5e1DF
    ------------------------------
    CBSNews 68
    Report: 14-year-old stabbed, killed after fight that started over sleepover https://t.co/soNqZ9Vf7w https://t.co/dkuCJRmNUp
    ------------------------------
    CBSNews 69
    Mattis slams China over weapons in South China Sea https://t.co/FZ19wnvhF8 https://t.co/sR4ucRIz9p
    ------------------------------
    CBSNews 70
    Three space station fliers head home, three prep for launch https://t.co/r6ic4cJGVQ https://t.co/kQOkGY6EN4
    ------------------------------
    CBSNews 71
    U.S. citizen killed in Nicaragua amid violence and social unrest https://t.co/BDA5RUt4zZ https://t.co/VmvMzf4V54
    ------------------------------
    CBSNews 72
    Who will pay for King Jong Un's hotel stay in Singapore? https://t.co/MRoeTLsXA3 https://t.co/hP9wslODtg
    ------------------------------
    CBSNews 73
    Pushing the envelope? Kim Jong Un's massive letter to President Trump https://t.co/y18wKCVUuv https://t.co/QV2CTVuLD1
    ------------------------------
    CBSNews 74
    Iowa judge agrees to halt fetal heartbeat abortion law https://t.co/dS87cm7Z7b https://t.co/FPdn1vJGcy
    ------------------------------
    CBSNews 75
    Small plane with 4 on board crashes off New York's Long Island, FAA says https://t.co/mNREjKjUmx https://t.co/jcmKQq2sfi
    ------------------------------
    CBSNews 76
    President Trump hints at longer path for North Korea to denuke https://t.co/rhd5rn8ffW https://t.co/0Mq4RDKtWU
    ------------------------------
    CBSNews 77
    Why are deaths among U.S. kids, teens on the rise? https://t.co/KiUl4c0z7s https://t.co/LWtm7VMh0n
    ------------------------------
    CBSNews 78
    The Dish: David Cáceres' recipes from La Panadería https://t.co/nKU7q8lK7v https://t.co/m7LmRu8io3
    ------------------------------
    CBSNews 79
    Shooting death of Steven Pitt, forensic psychiatrist, linked to killings of two women https://t.co/SG6nPJyZby https://t.co/ZjGr7vEVpv
    ------------------------------
    CBSNews 80
    Report: In memo to Mueller, Trump's lawyers argue he could not have obstructed justice https://t.co/zQxQVTbL7x https://t.co/qdHMse4zt6
    ------------------------------
    CBSNews 81
    Four killed, seven injured in Oregon highway crash, police say https://t.co/8zsJyRpsXV https://t.co/Hd3PLRhqzo
    ------------------------------
    CBSNews 82
    Brush fire forces thousands to evacuate parts of Southern California https://t.co/71zNMZYNu9
    ------------------------------
    CBSNews 83
    Police investigate deadly hit-and-run at baseball game in Maine https://t.co/V7lc3h2iCz https://t.co/iKEqTo9HGB
    ------------------------------
    CBSNews 84
    Search continues for 2 teens believed to have drowned in Georgia https://t.co/lCRqkWmD1f https://t.co/QNgZ0GvY6z
    ------------------------------
    CBSNews 85
    Report: 14-year-old stabbed, killed after fight that started over sleepover https://t.co/fBlpIdDAVu https://t.co/WJsQ804tvT
    ------------------------------
    CBSNews 86
    Chicago schools failed to protect students sexually abused by employees, report says https://t.co/B1uFRtT2MO https://t.co/qhCBNKI8TH
    ------------------------------
    CBSNews 87
    Sheriff: Suspect in disappearance of Fla. mom has had 18 aliases, lived in 33 cities https://t.co/Sck0OybMEg https://t.co/1aWqD5lzBE
    ------------------------------
    CBSNews 88
    Benedict Cumberbatch praised for tackling mugger https://t.co/j6NLmaAJWO https://t.co/dh9dV8sh7Q
    ------------------------------
    CBSNews 89
    Who will pay for King Jong Un's hotel stay in Singapore? https://t.co/MRoeTLsXA3 https://t.co/db6iTcti3K
    ------------------------------
    CBSNews 90
    U.S. citizen killed in Nicaragua amid violence and social unrest https://t.co/6EsQdBLSa8 https://t.co/vAOVvyB8T5
    ------------------------------
    CBSNews 91
    Santa Fe High School students graduate two weeks after deadly mass shooting https://t.co/hubWCfgeHj https://t.co/LJBOMPJNfM
    ------------------------------
    CBSNews 92
    Small plane with 4 on board crashes off New York's Long Island, FAA says https://t.co/mNREjKjUmx https://t.co/ihrVzsE2yO
    ------------------------------
    CBSNews 93
    President Trump hints at longer path for North Korea to denuke https://t.co/rhd5rn8ffW https://t.co/mKcpss9fH1
    ------------------------------
    CBSNews 94
    Attorney: $4 verdict meant to "punish" family of man shot by police in his garage https://t.co/IW0kmZ60cC https://t.co/L6YR53LSm6
    ------------------------------
    CBSNews 95
    Officials report a hang glider was killed in an accident at the "Gunfighter Skies Air &amp; Space Celebration"… https://t.co/4HmbdoRybM
    ------------------------------
    CBSNews 96
    A driver faces manslaughter charges after she struck and killed a 68-year-old man at a Maine baseball field.… https://t.co/h4Eb9D84JX
    ------------------------------
    CBSNews 97
    Two climbers dead after falling from Yosemite's El Capitan https://t.co/blwizkRHWp https://t.co/oTdJMkWXM7
    ------------------------------
    CBSNews 98
    A confidential letter obtained by The New York Times shows President Trump's personal lawyers making the case to sp… https://t.co/5UhnjzeHBR
    ------------------------------
    CBSNews 99
    "For children in the U.S., the rate of deaths has increased recently, and it's from a multitude of these injury dea… https://t.co/niM08stIrU
    ------------------------------
    CNN 0
    Actor Benedict Cumberbatch saves cyclist from muggers https://t.co/D0zALzFoCY https://t.co/hnQTowlGH9
    ------------------------------
    CNN 1
    Former New Jersey Gov. Chris Christie says it's an "outrageous claim" that President Trump can't obstruct justice… https://t.co/A1Xlidjyzk
    ------------------------------
    CNN 2
    Nearly 100 women suffered under dancing doctor's scalpel, lawyer says https://t.co/JKCekjz8bw https://t.co/4q69tZuiu1
    ------------------------------
    CNN 3
    When he was a kid, Richard Jenkins raised his hand in class so often bullies started calling him "Harvard."
    
    Now, a… https://t.co/okoYkkhJdn
    ------------------------------
    CNN 4
    Author Germaine Greer's dangerous ideas about rape https://t.co/cPrlwOCoPW (via @CNNOpinion) https://t.co/3vhDQGV5GJ
    ------------------------------
    CNN 5
    Sen. Bob Corker says GOP senators are working on plan to "push back" on Trump policy https://t.co/fx2wMRSMWZ https://t.co/xRiqlqwL7C
    ------------------------------
    CNN 6
    President Trump says Senate Democrats are "resisting the will" of voters https://t.co/E4qbFMRLPe https://t.co/7QE2uCdt7R
    ------------------------------
    CNN 7
    This FBI agent lost his gun during a dance-floor backflip and accidentally shot a bar patron https://t.co/ICK3IAFrgs https://t.co/cTfVCE5v9m
    ------------------------------
    CNN 8
    Gaza militants fire rockets towards Israel, and Israel Defense Forces responds with airstrikes… https://t.co/jStVpnYZly
    ------------------------------
    CNN 9
    Dog dies during Delta Air Lines layover in Michigan https://t.co/vmHsRHK4xU https://t.co/QdyLmeQ0u0
    ------------------------------
    CNN 10
    Is President Trump giving North Korea a pass on nukes? https://t.co/4jzsC63v3e (via @CNNOpinion) https://t.co/8FRCEkoVAh
    ------------------------------
    CNN 11
    Ben Maher takes St Tropez Global Champions Tour title https://t.co/ELTEABvr3E https://t.co/0oCYM8Zxfq
    ------------------------------
    CNN 12
    No sanctions relief without steps to denuclearize, US Defense Secretary James Mattis tells North Korea… https://t.co/IVl6Za5v2Z
    ------------------------------
    CNN 13
    Former President Bill Clinton says an impeachment process over the Russia probe would be underway if a Democrat wer… https://t.co/znmtIDv8nr
    ------------------------------
    CNN 14
    Silent forms of protest make bold statements at Parkland graduation https://t.co/nM71NgD7PO https://t.co/HIUt6o4qaA
    ------------------------------
    CNN 15
    Kansas Republican gubernatorial candidate Kris Kobach criticized for riding in a parade with a replica gun… https://t.co/jcbwKUMOJa
    ------------------------------
    CNN 16
    Jimmy Fallon crashes Marjory Stoneman Douglas graduation https://t.co/hOq6DiF2Fu https://t.co/hDqlcJtxJt
    ------------------------------
    CNN 17
    Thousands of acres ablaze in California, Colorado and New Mexico https://t.co/6nNgmuzJiS https://t.co/tQ0yCa6GoF
    ------------------------------
    CNN 18
    President Trump's lawyer, Rudy Giuliani, says Trump shouldn't testify because "our recollections keep changing"… https://t.co/mxFcjU5YD3
    ------------------------------
    CNN 19
    A solemn graduation for Marjory Stoneman Douglas seniors https://t.co/NqcmGRbxFc https://t.co/pJlnA5slXv
    ------------------------------
    CNN 20
    Ford dropped cars but it's keeping the Mustang. Here's why: https://t.co/gWdeLfWmch https://t.co/gHvz0gs8jr
    ------------------------------
    CNN 21
    Vatican replaces archbishop convicted of concealing child sex abuse https://t.co/NaFmQHVdN1 https://t.co/q0y3QV8Awp
    ------------------------------
    CNN 22
    Even the race to replace "Putin's favorite Congressman" unlikely to hinge on Russia https://t.co/EZUt9qXrLY https://t.co/2o1HuDUTFk
    ------------------------------
    CNN 23
    President Trump to host Ramadan dinner https://t.co/ZNoR75CrRA https://t.co/P1X4g2FQ7F
    ------------------------------
    CNN 24
    Hamptons "builder to the stars" among those feared dead in plane crash off Long Island https://t.co/aTSXQjz7L7 https://t.co/ZYiANx7mrD
    ------------------------------
    CNN 25
    Kuwait wants to foster more private sector firms as part of sweeping diversification plans https://t.co/tTy6dyJnR6 https://t.co/WvNLE1DXfe
    ------------------------------
    CNN 26
    Trump lawyers say he "dictated" statement on Trump Tower meeting, contradicting past denials https://t.co/vlFcqx1uiM https://t.co/14C2HmISbW
    ------------------------------
    CNN 27
    Trade wars are scary. Why isn't Wall Street freaking out? https://t.co/DFElDvMhKZ https://t.co/QWAZEMnoSj
    ------------------------------
    CNN 28
    Residents stranded with no power and water in area cut off by lava in Hawaii https://t.co/pxQjUKmjKk https://t.co/MJbm9rllIJ
    ------------------------------
    CNN 29
    What to expect at Monday's Apple event https://t.co/HXt7nltvOW https://t.co/TbQHmijp2P
    ------------------------------
    CNN 30
    Looking for lower Medicare drug costs? Ask your pharmacist for the cash price https://t.co/81QmKI8Xnm https://t.co/P3vUBOYPvJ
    ------------------------------
    CNN 31
    Former US attorney Preet Bharara says President Trump pardoning himself would be "almost self-executing impeachment… https://t.co/YhJVsmthKn
    ------------------------------
    CNN 32
    Canadian official Chrystia Freeland calls President Trump's tariffs insulting https://t.co/2JonOUzB4s https://t.co/mKSO8N04SK
    ------------------------------
    CNN 33
    How one month reshaped the US immigration landscape https://t.co/b3DFeEC3db https://t.co/YhNFc9UFpC
    ------------------------------
    CNN 34
    "The Americans" made motherhood the ultimate disguise https://t.co/jrF6Wjj933 (via @CNNopinion) https://t.co/2K1MMwKEjk
    ------------------------------
    CNN 35
    "Solo: A Star Wars Story" sinks 65% in second weekend https://t.co/xyeQpChuu2 https://t.co/kp2Md6qfxx
    ------------------------------
    CNN 36
    The New York Times highlights EPA Administrator Scott Pruitt's cozy relationship with a coal baron… https://t.co/VUv05naQ8O
    ------------------------------
    CNN 37
    "That's not what the American people, I think, would be able to stand for." Former US attorney Preet Bharara says i… https://t.co/InFFVlGCaA
    ------------------------------
    CNN 38
    Samantha Bee will address vulgar comment about Ivanka Trump on her show https://t.co/OyIbHMShHx https://t.co/KZvpq2sW3R
    ------------------------------
    CNN 39
    FBI agent loses his gun during dance-floor backflip, accidentally shoots bar patron https://t.co/F61P7oVf0C https://t.co/kaCXlPkI9r
    ------------------------------
    CNN 40
    Southwest Airlines comes under fire after an agent asks a mom to "prove" biracial child is hers… https://t.co/BTjEHSyG5H
    ------------------------------
    CNN 41
    The US Geological Survey has warned people not to roast marshmallows over volcanic vents in Hawaii https://t.co/j8bf0QdcBk
    ------------------------------
    CNN 42
    Vermont has a new law that will pay workers $10,000 to move there and work remotely https://t.co/BOsRh6adek https://t.co/BafSsgoD5f
    ------------------------------
    CNN 43
    Parents of Parkland victims are outraged about a new video game that would let players shoot up a school… https://t.co/7cLz5FT10V
    ------------------------------
    CNN 44
    More than 120 pregnant whales were killed during Japan's annual hunt last summer, prompting outrage among conservat… https://t.co/AMpERnhYGv
    ------------------------------
    CNN 45
    When he was a kid, Richard Jenkins raised his hand in class so often bullies started calling him "Harvard."
    
    Now, a… https://t.co/0prHEtH5ZV
    ------------------------------
    CNN 46
    These crosses represent each victim killed in the Santa Fe High School shooting. Retired carpenter Greg Zanis has b… https://t.co/mkX59AneVF
    ------------------------------
    CNN 47
    A Catholic school told its valedictorian his speech was too political. So he gave it outside on a bullhorn.… https://t.co/FIG7Jh43zR
    ------------------------------
    CNN 48
    When he was a kid, Richard Jenkins raised his hand in class so often bullies started calling him "Harvard."
    
    Now, a… https://t.co/q25qpxE052
    ------------------------------
    CNN 49
    "Chris, we've got your baby girl": Army Spc. Christopher M. Harris died serving in Afghanistan, just days after lea… https://t.co/RrOAxDYdWC
    ------------------------------
    CNN 50
    "Don't forget to show love": In a superhero cape, he feeds the hungry and homeless in Birmingham, Alabama. And he's… https://t.co/WZm3BWvrbN
    ------------------------------
    CNN 51
    Roseanne Barr says she begged ABC executives to keep her show on the air before the network canceled its hit sitcom… https://t.co/FcCBIrH65R
    ------------------------------
    CNN 52
    Alabama school administrators, who get approval from local authorities and go through training, will be permitted t… https://t.co/lYSarBzcgW
    ------------------------------
    CNN 53
    "Black lives don't matter," lawyer says after jury awards $4 in police killing https://t.co/DJTzgOXbfE https://t.co/nuBDS1RJKL
    ------------------------------
    CNN 54
    Only one more state needs to pass the Equal Rights Amendment to finally get it ratified https://t.co/y35AIYdJyp https://t.co/fc0zvM7b5G
    ------------------------------
    CNN 55
    Televangelist Jesse Duplantis says God spoke to him and wants him to get a new private jet that can cost up to $54… https://t.co/ZWQGuPcxiD
    ------------------------------
    CNN 56
    He was an NFL player. Now he's a Juilliard-trained opera singer. https://t.co/ItAsUVK7mh https://t.co/Eo1zyElF2q
    ------------------------------
    CNN 57
    This Chick-fil-A franchise owner is raising wages at his store to $17 an hour https://t.co/gjTVQnQzPL https://t.co/b8pVRyrNee
    ------------------------------
    CNN 58
    A high school football player who takes cannabis oil to prevent his seizures has been ruled ineligible to play in c… https://t.co/dxQvn0Zqqg
    ------------------------------
    CNN 59
    Ivanka Trump has been granted seven new trademarks in China as her father continues trade talks with Beijing -- and… https://t.co/KU3eraKgNT
    ------------------------------
    CNN 60
    Syrian President is to meet Kim Jong Un in North Korea, report says https://t.co/b6HeqI6RJG
    ------------------------------
    CNN 61
    Becky McCabe got down on one knee to propose to her girlfriend, Jessa Gillaspie. Jessa shrieked and immediately rea… https://t.co/eO2lkzyzCd
    ------------------------------
    CNN 62
    A high school football player who takes cannabis oil to prevent his seizures has been ruled ineligible to play in c… https://t.co/vI9KwkNwA1
    ------------------------------
    CNN 63
    This 8-year-old Georgia boy was with his mom when he saw an elderly woman struggling to climb the stairs with her w… https://t.co/UoGLZU54Kj
    ------------------------------
    CNN 64
    A teacher in Georgia responded the only way she knew how after receiving a letter from President Trump filled with… https://t.co/y07WzGq2JC
    ------------------------------
    CNN 65
    In 1997, a 6th-grade teacher wrote "Invite me to your Harvard graduation!" on a student's report card. Twenty-one y… https://t.co/S9wB18o8ox
    ------------------------------
    CNN 66
    A soldier died before meeting his baby, so his Army family welcomed her with open arms https://t.co/8sYSpYz1Yk https://t.co/bS0LWwOxeR
    ------------------------------
    CNN 67
    A man went to a salon and learned to do his wife's hair after she suffered a stroke and was unable to style it hers… https://t.co/tiYCyEPUOz
    ------------------------------
    CNN 68
    Rapper Falz's "This is Nigeria" video holds up a mirror for the country https://t.co/xdmgSkxpOO https://t.co/U1d6dzvPMY
    ------------------------------
    CNN 69
    A young Malian migrant who rescued a child dangling from a balcony will be made a French citizen and has been offer… https://t.co/atbtW5sNuU
    ------------------------------
    CNN 70
    North Korea will not get any sanctions relief until it has demonstrated "verifiable and irreversible" steps to denu… https://t.co/VaNkJqq3yH
    ------------------------------
    CNN 71
    Iraq War veteran Pat Ryan is running for Congress. He’s calling for an assault weapons ban: “The weapons that I car… https://t.co/MIcdUn6IJk
    ------------------------------
    CNN 72
    South Korean boy band @BTS_twt has become the first K-pop group to top the US Billboard 200 chart, with the debut o… https://t.co/e4pbDzVIH1
    ------------------------------
    CNN 73
    Archaeologists working at the ancient Roman city of Pompeii, Italy, uncovered the remains of a 30-year-old man who… https://t.co/Al0C75DN9F
    ------------------------------
    CNN 74
    NFL star JJ Watt has been awarded an honorary doctorate degree for raising over $37 million to help with recovery e… https://t.co/BTUkyeC2Id
    ------------------------------
    CNN 75
    Proceeds from economist's Nobel medal donated to non-profit https://t.co/kX9n19xQSv https://t.co/EFoIzSC9LV
    ------------------------------
    CNN 76
    Republican Sen. John McCain says he supports effort to force immigration bill to House floor https://t.co/16TqOGGc9Y https://t.co/9tRcu1XvXe
    ------------------------------
    CNN 77
    The largest wildfire in California's modern history is finally out, more than 6 months after it started… https://t.co/ghBrwkNqi1
    ------------------------------
    CNN 78
    A dog being flown by Delta Air Lines from Phoenix to Newark, New Jersey, died during a layover in Detroit, the airl… https://t.co/CIZZl7NzWT
    ------------------------------
    CNN 79
    Why startup visas are good for America https://t.co/jlC9GmqkEK (via @CNNOpinion) https://t.co/leqIYx9wu2
    ------------------------------
    CNN 80
    Four killings, including one of prominent psychologist, in Arizona are likely connected, police say… https://t.co/bN8Wy37Uzp
    ------------------------------
    CNN 81
    Virginia GOP chooses replacement for 5th district nominee https://t.co/aidAnQaUJT https://t.co/40fjghIVmj
    ------------------------------
    CNN 82
    Valerie Jarrett, former top aide to President Obama, makes an appeal to get out the vote in 2018, saying "it's time… https://t.co/kOuEcwmX9X
    ------------------------------
    CNN 83
    "The Big Bang Theory" star Jim Parsons tells @VanJones68 he thought about Roseanne Barr's racist tweet "on an emoti… https://t.co/DZ5mZaqIkP
    ------------------------------
    CNN 84
    Kenyan President orders top officials to take lie-detector tests in corruption crackdown https://t.co/jaKPXboTt1 https://t.co/q5lY5cX0jq
    ------------------------------
    CNN 85
    An attorney representing three women in cases against a doctor known for singing and dancing while performing surge… https://t.co/Dog9UV2r2w
    ------------------------------
    CNN 86
    "The crew, the people you don't see, the people you don't know — there are so many people employed..." Actor Jim Pa… https://t.co/hlRURomCmj
    ------------------------------
    CNN 87
    Five Star Movement supporters rally in Rome to celebrate new government https://t.co/TG0G0PfQ8Q https://t.co/khOs2BwJFL
    ------------------------------
    CNN 88
    The 30-year-old man who was still living with his parents says that he finally moved out, 10 days after a judge ord… https://t.co/SJnu3AdCmf
    ------------------------------
    CNN 89
    "The Big Bang Theory" star Jim Parsons talks about the fallout surrounding comedian Roseanne Barr's racist tweet, w… https://t.co/V2jEahrqYU
    ------------------------------
    CNN 90
    Six major allies blast US over tariffs https://t.co/spgQbkUKR7 https://t.co/XqBSYXUalE
    ------------------------------
    CNN 91
    This is what it's like to surf near the Korean Demilitarized Zone https://t.co/8yu12Jk0Uv https://t.co/M3PaYFs75Q
    ------------------------------
    CNN 92
    Two climbers fell to their deaths Saturday at Yosemite National Park in California, a park spokeswoman said… https://t.co/jaQW500EV1
    ------------------------------
    CNN 93
    This 8-year-old Georgia boy was with his mom when he saw an elderly woman struggling to climb the stairs with her w… https://t.co/C2z0TDF8zu
    ------------------------------
    CNN 94
    Fifth federal judge rules HHS unlawfully ended teen pregnancy prevention grant funding https://t.co/FbAHxbgMmb https://t.co/p7iUsaxqQ6
    ------------------------------
    CNN 95
    Serena Williams to face Maria Sharapova at French Open https://t.co/sVQ0mY8OZ0 https://t.co/hQAbDfu8Lb
    ------------------------------
    CNN 96
    What aluminum and steel tariffs will mean to you https://t.co/nyv92aLySO https://t.co/rSFOTHUraK
    ------------------------------
    CNN 97
    Outgoing Missouri Gov. Eric Greitens, accused of revenge porn, signs law criminalizing it https://t.co/VtbQzEt6vW https://t.co/yMKjrU1vg5
    ------------------------------
    CNN 98
    US customs seizes Ohio family's life savings at airport https://t.co/rI48TGrwvr https://t.co/6VarTheMrf
    ------------------------------
    CNN 99
    White House plans to nominate conservative documentarian, Bannon ally, to lead government media agency… https://t.co/dfPoET7g4V
    ------------------------------
    FoxNews 0
    .@AntonioSabatoJr: "There's a difference between legal and illegal immigration." @NextRevFNC https://t.co/97nm2kxvIo
    ------------------------------
    FoxNews 1
    .@KennedyNation on Russia probe: "I think John Brennan, like James @Comey, they think they're saviors of these gild… https://t.co/BRe4zHXYfq
    ------------------------------
    FoxNews 2
    .@KennedyNation: "There's always a disconnect between the way Washington sees the economy and the way families see… https://t.co/fMdKTR67GY
    ------------------------------
    FoxNews 3
    High school biology teacher, 34, accused of having sex with student https://t.co/1TNSKr6P6S
    ------------------------------
    FoxNews 4
    Portland sees bloody fighting as Antifa activists storm Patriot Prayer rally https://t.co/KUondrTvuy
    ------------------------------
    FoxNews 5
    Patriotic Business Pushes Back After Town Tells It to Remove 'Excessive' American Flags https://t.co/xdQLQoXMUI
    ------------------------------
    FoxNews 6
    OPINION: Here are ten ways to pray for our high school graduates  https://t.co/RNcYMNtV7K
    ------------------------------
    FoxNews 7
    Dallas woman says she killed husband for beating their pet cat, cops say https://t.co/xOMiV6fqIb
    ------------------------------
    FoxNews 8
    'Fight Club' author learns big reason why he's broke, apologizes https://t.co/1Ex6FK8BBZ
    ------------------------------
    FoxNews 9
    'Solo: A Star Wars Story' plummeting at the box office https://t.co/9ANVkdsv65
    ------------------------------
    FoxNews 10
    RT @FoxNewsResearch: African-American Unemployment Rate:
    •May 2010: 15.5%
    •May 2011: 16.3%
    •May 2012: 13.5%
    •May 2013: 13.4%
    •May 2014: 11.…
    ------------------------------
    FoxNews 11
    RT @FoxNewsResearch: U.S. Share of Global Military Spending
    —2017—
    •Ranks as world's top spender
    •$610 billion, 35% of global share
    •Down f…
    ------------------------------
    FoxNews 12
    Legal bare-knuckle boxing kicks off in Wyoming https://t.co/GVeQhrZmC5
    ------------------------------
    FoxNews 13
    Guatemala volcano kills at least 6, covers villages in ash https://t.co/P8GiPf4XJS
    ------------------------------
    FoxNews 14
    .@larry_kudlow on NK summit: "The key point is that we're sitting down. And the second key point is that the presid… https://t.co/RB2obq3U66
    ------------------------------
    FoxNews 15
    .@larry_kudlow on tariffs: "If you don't have a level playing field, you can't operate free trade." #FoxNewsSunday… https://t.co/xaRSsCUxi4
    ------------------------------
    FoxNews 16
    MONDAY: @PressSec Sarah Sanders and @MarkLevinShow join @SeanHannity at 9p ET. Tune in! #Hannity https://t.co/KRHjcUfAdo
    ------------------------------
    FoxNews 17
    .@Lavarbigballer has a message for @NFL players who disapprove of the national anthem policy — stand up or get out.… https://t.co/bvsrVr1XAc
    ------------------------------
    FoxNews 18
    .@CLewandowski_: "@POTUS has offered to sit down with the Mueller investigators but he has said...they want to unde… https://t.co/pmHg7lWqdF
    ------------------------------
    FoxNews 19
    Despair, hope and drugs in America: Is anyone safe? https://t.co/efhC6Z0YJx
    ------------------------------
    FoxNews 20
    Alabama death row inmate apparently commits suicide by hanging himself in jail cell https://t.co/cUmLPCZfvj
    ------------------------------
    FoxNews 21
    Bill Clinton says impeachment hearings would have begun already if a Democrat were president  https://t.co/TWY3zf1R6R
    ------------------------------
    FoxNews 22
    #RoyalWedding expert 'Thomas J. Mace-Archer-Mills' is actually 'Tommy' from NY, reports say https://t.co/Lc2etr4Y3T
    ------------------------------
    FoxNews 23
    .@jimmyfallon gives surprise speech at Stoneman Douglas graduation https://t.co/B1x3Xe3gAQ
    ------------------------------
    FoxNews 24
    Michael Goodwin: James Comey isn't above the law https://t.co/kxnW3mn3t1
    ------------------------------
    FoxNews 25
    Sales for @MariahCarey's Vegas shows are 'a disaster'  https://t.co/eakfBAynqA
    ------------------------------
    FoxNews 26
    .@dangainor: Roseanne Barr is off TV but Samantha Bee is still on -- just another example of the media’s double sta… https://t.co/k0zJOYKFBV
    ------------------------------
    FoxNews 27
    .@JonHuntsman: "@POTUS has stated before that he would like to get together with President Putin at some point and… https://t.co/vFLvPFl61l
    ------------------------------
    FoxNews 28
    A young girl couldn't wait to greet her relative from the 4th Infantry Brigade Combat Team (Airborne), 25th Infantr… https://t.co/YyaxfO0Swb
    ------------------------------
    FoxNews 29
    Peggy Noonan: America’s lost faith in itself. We can bring it back https://t.co/GToL36Lo1U
    ------------------------------
    FoxNews 30
    .@JonHuntsman: "Every foreign policy pundit and analyst who would've been asked a few years ago never would've pred… https://t.co/aHYUQ9LjRl
    ------------------------------
    FoxNews 31
    .@SpeakerBoehner: "There is no Republican Party. There's a Trump Party. The Republican Party is kind of taking a na… https://t.co/t6BB7dX92H
    ------------------------------
    FoxNews 32
    LaVar Ball to NFL Protesters: Stand for the National Anthem or 'Get Out' https://t.co/AYm4GvFKj1
    ------------------------------
    FoxNews 33
    .@newtgingrich: "I would predict today we're closer to a red wave than a #bluewave in terms of the fall campaign." https://t.co/9zNrDbRLrz
    ------------------------------
    FoxNews 34
    .@MariaBartiromo: "We're talking about almost 5% economic growth - there is NOTHING that @NancyPelosi just said tha… https://t.co/J8nyvBfZhG
    ------------------------------
    FoxNews 35
    ‘Despair and Hope in America: Fox News Investigates’ – Watch this preview and don't miss part one of this special s… https://t.co/UEGUdqxe7n
    ------------------------------
    FoxNews 36
    .@BillClinton Pushes Back on Gillibrand Saying He Should've Resigned Over Lewinsky Scandal 
     https://t.co/4kE4PRGO5Y
    ------------------------------
    FoxNews 37
    Three stranded in Kilauea lava zone are airlifted; cops cite 7 others for loitering near lava https://t.co/1g23BSVdGA
    ------------------------------
    FoxNews 38
    On "Justice," Anthony @Scaramucci reflected on the controversial comments made by Samantha Bee against @IvankaTrump… https://t.co/YTvgKpD0gr
    ------------------------------
    FoxNews 39
    Jonathan Wachtel on North Korea: "We could have a move toward having nuclear energy as an option for them."… https://t.co/fHiCDcmtP4
    ------------------------------
    FoxNews 40
    Jonathan Wachtel: "The nuclear arsenal that Kim Jong Un has now is his life insurance." https://t.co/PZcpa5sWSO https://t.co/w9jziej6ct
    ------------------------------
    FoxNews 41
    Prostitutes and British tourists in violent clashes at notorious party resort in Spain https://t.co/wvZY80yD3o
    ------------------------------
    FoxNews 42
    Suspect arrested after gunfire heard near San Diego marathon; police say no ongoing threat https://t.co/osWYIuwc6H https://t.co/cAYUcCVzhM
    ------------------------------
    FoxNews 43
    Despite drop-ins by Alec Baldwin, Robert De Niro, Mississippi election season sees unexpected star: civility https://t.co/Dwg6cI0ReS
    ------------------------------
    FoxNews 44
    .@DineshDSouza: "My favorite term is the 'double jeopardy loophole' - so this basic Constitutional protection, Ed,… https://t.co/mvQ27GnIkG
    ------------------------------
    FoxNews 45
    BREAKING: Suspect arrested after shots heard near finish line of San Diego marathon https://t.co/iNx7E3VwAY
    ------------------------------
    FoxNews 46
    Maine police rescue a dozen ducklings from sewer grate https://t.co/mHTl5sJj1x
    ------------------------------
    FoxNews 47
    Remains of 8 veterans, long unclaimed, buried in San Antonio  https://t.co/1K6hmW1iaP
    ------------------------------
    FoxNews 48
    Many breast cancer patients can skip chemo, big study finds https://t.co/Ybfe9SItCf
    ------------------------------
    FoxNews 49
    Dallas woman says she killed husband for beating their pet cat, cops say https://t.co/ifDwufNs4s
    ------------------------------
    FoxNews 50
    Peter Navarro: "No president has fought the war on poverty better than Donald J. Trump." https://t.co/ANE9uICCZl
    ------------------------------
    FoxNews 51
    San Diego marathon scare: Gunfire reported from rooftop overlooking race https://t.co/iNx7E3VwAY
    ------------------------------
    FoxNews 52
    On "Justice," @RealCandaceO told @JudgeJeanine how difficult it is to be a conservative on campus today.… https://t.co/OmHTQgUTpk
    ------------------------------
    FoxNews 53
    Police hunt murder suspect in Connecticut mom's killing https://t.co/ERPKCcBc3d
    ------------------------------
    FoxNews 54
    .@brithume on NK summit: "I think the president's seeming eagerness for all of this to go forward is a little troub… https://t.co/BQgWhWVAIc
    ------------------------------
    FoxNews 55
    .@larry_kudlow: "In the world trade game, rule breaking is all over the place. That's why I think that the presiden… https://t.co/O08Fq0HcML
    ------------------------------
    FoxNews 56
    May jobs report. #FoxNewsSunday https://t.co/XIIaQOe6FA
    ------------------------------
    FoxNews 57
    .@larry_kudlow on tariffs: "The NAFTA talks haven't broken down. We're still having those conversations. And we're… https://t.co/lpIf0oleor
    ------------------------------
    FoxNews 58
    President @realDonaldTrump's tariffs. #FoxNewsSunday https://t.co/3VOpeAr4uq
    ------------------------------
    FoxNews 59
    .@larry_kudlow on tariffs: "if you don't have a level playing field, you can't operate free trade." #FoxNewsSunday… https://t.co/AKp6N4lKH0
    ------------------------------
    FoxNews 60
    .@newtgingrich: We're Closer to a 'Red Wave' Than a 'Blue Wave' in November https://t.co/LrJ2rP94jo
    ------------------------------
    FoxNews 61
    .@larry_kudlow on tariffs: "The World Trade Organization, which sets these rules, has been totally ineffectual. It… https://t.co/4J0aWxCQp3
    ------------------------------
    FoxNews 62
    .@larry_kudlow on tariff retaliation threats: "This is a trade dispute, if you will. It can be solved if people wor… https://t.co/3PpUplTxAM
    ------------------------------
    FoxNews 63
    .@larry_kudlow on tariffs: "[@POTUS] has declared our steel industry a national security matter and he hopes throug… https://t.co/AcOaSsJIL5
    ------------------------------
    FoxNews 64
    @larry_kudlow on NK summit: "Now, as we head into the negotiations, I think the president is being very realistic a… https://t.co/KuXXjYC7bW
    ------------------------------
    FoxNews 65
    After Roseanne's racist tweet, Obama ally Valerie Jarrett starts wooing voters https://t.co/NWJbG0Oaax
    ------------------------------
    FoxNews 66
    MONDAY: @PressSec Sarah Sanders and @MarkLevinShow join @SeanHannity at 9p ET. Tune in! #Hannity https://t.co/rHd4sahcys
    ------------------------------
    FoxNews 67
    .@larry_kudlow on NK summit: "The key point is that we're sitting down. And the second key point is that the presid… https://t.co/AyXMO2idNl
    ------------------------------
    FoxNews 68
    On "Sunday Morning Futures," Rep. @DevinNunes, a Republican from California, weighed on in Google, which came under… https://t.co/xx73c5i32z
    ------------------------------
    FoxNews 69
    Dog found dead in crate during Delta Air Lines layover for cross-country flight https://t.co/a3noVwIITC
    ------------------------------
    FoxNews 70
    .@CLewandowski_ on @POTUS' pardons: "Under the Constitution, he has the legal authority to pardon anybody. That's a… https://t.co/vIIhFd7asa
    ------------------------------
    FoxNews 71
    .@CLewandowski_: "[President @realDonaldTrump] clearly respects the rule of law in the country. There's no question… https://t.co/xR7ef4FAKJ
    ------------------------------
    FoxNews 72
    .@CLewandowski_: "@POTUS has offered to sit down with the Mueller investigators but he has said...they want to unde… https://t.co/l6KMXxyPmZ
    ------------------------------
    FoxNews 73
    ‘Despair and Hope in America: Fox News Investigates’ – Don't miss part one of this special series on ‘Fox Report,’… https://t.co/mcfNj4pofE
    ------------------------------
    FoxNews 74
    Idaho teacher accused of feeding live puppy to turtle faces charge https://t.co/EpND6hOisM
    ------------------------------
    FoxNews 75
    Robert Mueller needs to 'man up,' avoid pulling 'another Comey' as Russia probe wraps up, Giuliani says https://t.co/fpgGFOr8hv
    ------------------------------
    FoxNews 76
    .@RealCandaceO: "You see so many students that are speaking out saying that their professors are upset with them fo… https://t.co/hiZ0QV4gSH
    ------------------------------
    FoxNews 77
    "Last month, 200,000 more black Americans had jobs than they had the month before. Absolutely remarkable."
    
    On… https://t.co/JbmpugFzFB
    ------------------------------
    FoxNews 78
    .@Scaramucci: "What hasn't been mentioned all week is that President Clinton also pardoned his brother Roger Clinto… https://t.co/NwjQAq0hCW
    ------------------------------
    FoxNews 79
    Georgia cop fired after hitting suspect fleeing on foot with his patrol car https://t.co/xUjCCiJr38
    ------------------------------
    FoxNews 80
    .@KellyannePolls: "One eighth of our U.S. Circuit Court now includes a Trump judge." https://t.co/QApBQwuHG8
    ------------------------------
    FoxNews 81
    .@Peggynoonannyc: America’s lost faith in itself. We can bring it back
    https://t.co/OKrBq90KYc
    ------------------------------
    FoxNews 82
    Woman, childhood best friend are revealed to be half-siblings https://t.co/WHVft0PrH0
    ------------------------------
    FoxNews 83
    TONIGHT: A brand new "Life, Liberty, and Levin" with @marklevinshow - Tune in at 10p ET on Fox News Channel! https://t.co/KcIyJeV36Z
    ------------------------------
    FoxNews 84
    .@cvpayne: "Last month, 76,000 blacks entered the labor force. The unemployment rate for black Americans went to an… https://t.co/0o5Fp0Ke7T
    ------------------------------
    FoxNews 85
    On @NextRevFNC, @SteveHiltonx investigates Gov. Jerry Brown's policies and their effect on California. Tune in toni… https://t.co/P2SpGanG8M
    ------------------------------
    FoxNews 86
    .@SpeakerBoehner: "There is no Republican Party. There's a Trump Party. The Republican Party is kind of taking a na… https://t.co/fr6L4MPCmb
    ------------------------------
    FoxNews 87
    'Builder to the stars,' his wife among 4 feared dead after plane crashes off New York coast https://t.co/ZoreukG5Yn
    ------------------------------
    FoxNews 88
    On "Sunday Morning Futures," @WhiteHouse Trade Policy Director Peter Navarro praised President @realDonaldTrump's e… https://t.co/H2hR2LtS9D
    ------------------------------
    FoxNews 89
    .@NancyPelosi hits bonuses, tax benefits as "crumbs." https://t.co/zNcp4a2nBn
    ------------------------------
    FoxNews 90
    The team you trust. The network you love.
    
    Watch 'Hannity,' weeknights at 9p ET on Fox News Channel. https://t.co/7tl5PucYOi
    ------------------------------
    FoxNews 91
    .@newtgingrich: "Menendez in NJ is in trouble because of his scandals. I think we're very likely to pick up Florida… https://t.co/BABYpgEew5
    ------------------------------
    FoxNews 92
    .@GovMikeHuckabee: "All we need to do is take [@NancyPelosi's] words, put it on television, and play it over and ov… https://t.co/ifefT3tVsb
    ------------------------------
    FoxNews 93
    TONIGHT: 'Legends &amp; Lies' Returns With a Riveting Look at John Wilkes Booth - Hosted by Brian @Kilmeade, it all beg… https://t.co/1WcXoiI59q
    ------------------------------
    FoxNews 94
    Indiana police officers support daughter of fallen trooper at graduation more than 10 years after his death https://t.co/8330yRYoPf
    ------------------------------
    FoxNews 95
    .@emilyjashinsky: "What I cannot wrap my head around is the fact that [@JoyAnnReid] pretty much clearly fabricated… https://t.co/4B96m11fFn
    ------------------------------
    FoxNews 96
    .@Richardafowler: "I don't think you can conflate the two cases." #MediaBuzz https://t.co/ENEkW7Slpw
    ------------------------------
    FoxNews 97
    'Comey and the boys' shouldn't have hidden probe of Paul Manafort during campaign, Trump says… https://t.co/0TzMdV3bDj
    ------------------------------
    FoxNews 98
    FBI agent accidentally fires gun while dancing at a Denver nightclub https://t.co/cHhoRe5nTe
    ------------------------------
    FoxNews 99
    .@KatiePavlich: "The mainstream media, let's not forget, propped @realDonaldTrump during the primary and as soon as… https://t.co/X4qik1e0HT
    ------------------------------
    nytimes 0
    The New York Times's summer reading list is here to browse:
     
    Thrillers
    Cooking 
    True Crime
    Movies &amp; TV
    Romance
    Tra… https://t.co/p8h7Wse551
    ------------------------------
    nytimes 1
    Slovenia Elections Tilt Another European Country to the Right https://t.co/TFunNGI03j
    ------------------------------
    nytimes 2
    RT @mikiebarb: Tomorrow we begin a powerful five-part, week-long series on The Daily -- like nothing we've done before. It's about a city a…
    ------------------------------
    nytimes 3
    RT @NYTSports: Once again, @TheSteinLine and @BenHoffmanNYT bring you their view of the NBA finals in real time. https://t.co/Ls56JcfMOU
    ------------------------------
    nytimes 4
    Why Is Trump ‘Not Important’ in Mexico Election? All Candidates Are Against Him https://t.co/2u10DZyzrv
    ------------------------------
    nytimes 5
    The New York Times review of "The President Is Missing," a thriller and escapist fairy tale co-written by Bill Clin… https://t.co/5Th4xtlMDL
    ------------------------------
    nytimes 6
    The Shame in Puerto Rico https://t.co/GfovgSUXIe
    ------------------------------
    nytimes 7
    Young adults are not only marrying and having children later in life than previous generations, but taking more tim… https://t.co/dO25Frw5jB
    ------------------------------
    nytimes 8
    3 women say they were led to believe the film mogul Harvey Weinstein’s criminal defense firm wanted to represent th… https://t.co/uPZ43CaPRH
    ------------------------------
    nytimes 9
    "When you’re not sitting across from someone, you’re sitting across from the world." The joys of a table for one in… https://t.co/qyojCy4Jbs
    ------------------------------
    nytimes 10
    On Golf: Tiger Woods Didn’t Win Again. But He Will. https://t.co/BeSbKDoH9m
    ------------------------------
    nytimes 11
    Show Us Your Wall: Andy Warhol in the Powder Room. Christopher Wool on the Floor. https://t.co/2wxCbQvsAb
    ------------------------------
    nytimes 12
    There are a few low-tech tricks that can make switching time zones and taking long haul flights a little easier https://t.co/e8pEcqppJu
    ------------------------------
    nytimes 13
    “If it was 5,000 dogs, there would be outrage. If it was 5,000 blonde-haired, blue-eyed women, there would be outra… https://t.co/rhAvTLU7Xx
    ------------------------------
    nytimes 14
    N.B.A. Finals 2018: Cavs vs. Warriors Game 2 Updates https://t.co/RwHYfYHo23
    ------------------------------
    nytimes 15
    A guilty pleasure for many children of the '90s: A personal pan pizza from Pizza Hut. What corporate delivery pie d… https://t.co/DHDflBQHX5
    ------------------------------
    nytimes 16
    Alone https://t.co/G3zzMesQUR
    ------------------------------
    nytimes 17
    A Soldier Died After Racist Hazing. Now His Story Is an Opera. https://t.co/7dHOsbvj6L
    ------------------------------
    nytimes 18
    Sutton Foster: "Some people are like, 'You sing?' I’m like, 'Yes, I sing.' It drives me crazy. It’s been fascinatin… https://t.co/LPMF0SbInb
    ------------------------------
    nytimes 19
    Democrats Hope an Asian Influx Will Help Turn Orange County Blue https://t.co/CbKFAPnsc7
    ------------------------------
    nytimes 20
    After an already rickety launch, “Solo: A Star Wars Story” took a nose-dive at the box office in its second week in… https://t.co/YB3U7nN12y
    ------------------------------
    nytimes 21
    The economy is in a sweet spot, with steady growth and broad improvement in the labor market https://t.co/lnbaMQMJn7
    ------------------------------
    nytimes 22
    RT @marclacey: As California prepares to vote Tuesday, may I introduce our stellar team of NYT national correspondents based there: @adamna…
    ------------------------------
    nytimes 23
    Here are some recommendations if you're wondering what to watch for the remainder of your weekend https://t.co/t90PQMmD9D
    ------------------------------
    nytimes 24
    “Something weird happened,” the president of the Yosemite Climbing Association said of the deadly fall. “There’s no… https://t.co/7ZRLy1P8FJ
    ------------------------------
    nytimes 25
    Michael Lewis is part of a growing group of A-list authors bypassing print and releasing audiobook originals, hopin… https://t.co/hom5zHaSrr
    ------------------------------
    nytimes 26
    “Here was Sherlock Holmes fighting off 4 attackers just round the corner from Baker Street.” https://t.co/RUl2gLJPNc
    ------------------------------
    nytimes 27
    Here are some ideas for what to cook this week https://t.co/haDa1gX5yS
    ------------------------------
    nytimes 28
    Nearly 4 years after the chokehold death of Eric Garner and vows of greater transparency, the disciplinary history… https://t.co/G0ckyXzyXR
    ------------------------------
    nytimes 29
    How Marc Jacobs fell out of fashion: He left Louis Vuitton, shrunk his business, shut most of his stores and lost h… https://t.co/wi2WV98nsj
    ------------------------------
    nytimes 30
    “I wanted to cry,” said the chief of orthopedic service at a hospital in Afghanistan, where 7 children were brought… https://t.co/oI62AFfbTw
    ------------------------------
    nytimes 31
    Disappear into a thriller, a romance, a cookbook or the great outdoors: Our summer reading guide has books for ever… https://t.co/uWhmLeklN3
    ------------------------------
    nytimes 32
    In the absence of a customs union, every single one of the thousands of trucks that pass through the British port a… https://t.co/ZRczelRCB9
    ------------------------------
    nytimes 33
    3 women say they were led to believe the film mogul Harvey Weinstein’s criminal defense firm wanted to represent th… https://t.co/9SKeoleTIZ
    ------------------------------
    nytimes 34
    "Kanye West still has a streak of compassion and empathy, in the rare moments when he’s not thinking only of himsel… https://t.co/rEeoa7jZ5w
    ------------------------------
    nytimes 35
    Planning the details of the Tump-Kim summit is very, very tricky: Who will enter the room first? Who will pay for t… https://t.co/safgULgK3j
    ------------------------------
    nytimes 36
    Security officials said a boat had been packed with about 180 migrants. Dozens drowned when the boat capsized. https://t.co/TDcJobhcb3
    ------------------------------
    nytimes 37
    Steven Pitt, a high-profile forensic psychiatrist, was shot dead on Thursday. Less than 24 hours later, 2 paralegal… https://t.co/Zy1uG2juvN
    ------------------------------
    nytimes 38
    RT @melbournecoal: The NYT is on the ground in districts across CA as we await the primary. I’ll be in CA-25, just north of LA, where 4 Dem…
    ------------------------------
    nytimes 39
    RT @Watching: Are you already mourning "The Americans"? Here's what to watch next: https://t.co/l1VuxJGS6k
    ------------------------------
    nytimes 40
    In Opinion
    
    Op-Ed contributor Steve Kettmann writes, "California doesn’t just oppose Mr. Trump; it offers a better… https://t.co/VPTf7Lv2JY
    ------------------------------
    nytimes 41
    RT @amyfiscus: Even when dealers are caught selling guns illegally, they routinely go unpunished by the ATF. @AliWatkins scoop https://t.co…
    ------------------------------
    nytimes 42
    It was Christmas Day when the 34-year-old woman acknowledged that something was really wrong. She realized she was… https://t.co/MFg0zYKT3x
    ------------------------------
    nytimes 43
    Use your day off to catch up on some great reads you may have missed https://t.co/5F6T7xur5r
    ------------------------------
    nytimes 44
    RT @marclacey: ¿Quedarse o partir? Dos puertorriqueñas frente a una decisión https://t.co/RbZu1aWLxe via @nytimesES
    ------------------------------
    nytimes 45
    Here are the week’s top stories, and a look ahead https://t.co/UR0Y41WK6r
    ------------------------------
    nytimes 46
    The London police have blamed a bleak rap genre, drill, for a rise in violence. They've gotten YouTube to take down… https://t.co/iR5NgPyLaF
    ------------------------------
    nytimes 47
    “If it was 5,000 dogs, there would be outrage. If it was 5,000 blonde-haired, blue-eyed women, there would be outra… https://t.co/79qhh12yVB
    ------------------------------
    nytimes 48
    In Opinion
    
    Op-Ed columnist @MaureenDowd writes that Obama "pushed aside his loyal vice president, who was consider… https://t.co/RRwfLvDAAj
    ------------------------------
    nytimes 49
    RT @NYTSports: The Warriors' explosive third-quarter runs have become a phenomenon, and nobody knows quite what to make of it. But a closer…
    ------------------------------
    nytimes 50
    On Saturday, Pedro Sánchez was sworn in as Spain’s new prime minister, a stunning and rapid turnaround for a man wh… https://t.co/jZSHt2632n
    ------------------------------
    nytimes 51
    “He told me nobody would help me, because I don’t have papers." Reports of domestic violence and sexual assault in… https://t.co/SLrX3XxChn
    ------------------------------
    nytimes 52
    Beijing officials refused to pledge any additional purchases from the United States without an American agreement t… https://t.co/gLfzlTqbVa
    ------------------------------
    nytimes 53
    If nothing else, Jon Tester is incautious, at least compared to most of the other Senate Democrats up for re-electi… https://t.co/1H8ThOZLrm
    ------------------------------
    nytimes 54
    President Bashar al-Assad of Syria plans to visit North Korea’s leader, Kim Jong-un, the North’s state-run news med… https://t.co/le82t3KpYc
    ------------------------------
    nytimes 55
    RT @marclacey: “Editor’s note: This article was published without a byline and photo credits to protect the journalist’s safety. “ https://…
    ------------------------------
    nytimes 56
    Widely different estimates of Hurricane Maria’s death toll in Puerto Rico have led to confusion. Here is a guide to… https://t.co/Ido3ZF11in
    ------------------------------
    nytimes 57
    Many women with breast cancer can skip chemotherapy, a study found. “We can spare thousands and thousands of women… https://t.co/TXDafLWE2D
    ------------------------------
    nytimes 58
    RT @marclacey: California is about to hold a key primary election that will affect the direction of the state, and the nation. To follow ou…
    ------------------------------
    nytimes 59
    A 20-page letter from President Trump's lawyers to Robert Mueller, obtained by The New York Times, argues that the… https://t.co/ybw1yuh4Md
    ------------------------------
    nytimes 60
    We invited a group of New York Times journalists who happen to possess the anatomy in question to pick apart the co… https://t.co/o9IMFDmzmK
    ------------------------------
    nytimes 61
    Assad Said to Plan Meeting With North Korea’s Leader https://t.co/Lfku9w7iWs
    ------------------------------
    nytimes 62
    Writers like Ada Calhoun and Michael Lewis are part of a growing group of A-list authors bypassing print and releas… https://t.co/vNfOz6tiwD
    ------------------------------
    nytimes 63
    Pedro Sánchez, Spain’s New Leader, Returns From the Political Wilderness https://t.co/BLmg6gtKqj
    ------------------------------
    nytimes 64
    Trump often points out that his predecessors left him the “mess” of a nuclear-armed North Korea — errors he vows no… https://t.co/xN3F5jDRN3
    ------------------------------
    nytimes 65
    The Californization of America https://t.co/Ic8wgrjwnB
    ------------------------------
    nytimes 66
    Vancouver real estate is so expensive that politicians want to tax it into submission, and many homeowners are OK w… https://t.co/WZ38GOGxhT
    ------------------------------
    nytimes 67
    Scott Pruitt, who has reversed Obama-era EPA rules, enjoys cozy ties with a coal baron who got him a superfan exper… https://t.co/lWerrCr42b
    ------------------------------
    nytimes 68
    Good News for Women With Breast Cancer: Many Don’t Need Chemo https://t.co/L9or3uc4Is
    ------------------------------
    nytimes 69
    This past week brought 2 developments that could prove pivotal in November for Senate Republicans  https://t.co/tLrn1nA5Mi
    ------------------------------
    nytimes 70
    The top candidates for California governor grapple with the popularity of Jerry Brown and their own indiscretions a… https://t.co/QXlrM8epcL
    ------------------------------
    nytimes 71
    Saudi Arabia Names New Ministers Close to Powerful Crown Prince https://t.co/E1QekZjpCE
    ------------------------------
    nytimes 72
    Encounters: This Harry Potter Uses a Bow and Arrow. Not a Wand. https://t.co/2KLyUDzJKw
    ------------------------------
    nytimes 73
    Sunday Routine: How Angela Goding, of MoMA PS1, Spends Her Sundays https://t.co/BDaXC9Lkdr
    ------------------------------
    nytimes 74
    Will a Canadian version of Donald Trump become leader of the country's most important province? https://t.co/i1gLYTDIsg
    ------------------------------
    nytimes 75
    Pedro Markun once thought hacking Brazil’s political system was the best way to change it. Now, he wants to do more… https://t.co/DIb3KpEtxn
    ------------------------------
    nytimes 76
    The author Malcolm Gladwell: "I would release 95% of prisoners if I had the chance" https://t.co/eSrFSbsoHc
    ------------------------------
    nytimes 77
    The work culture for families has been stubbornly slow to change in Washington. Meet the federal regulator who brin… https://t.co/ln0ZRVntC8
    ------------------------------
    nytimes 78
    The gear you’ll need to survive in the wilderness. (Or your backyard.) https://t.co/AUQEtB08Im
    ------------------------------
    nytimes 79
    Trilobites: Ladybugs, Aphids and the Toxic Combat That Might Be Happening in Your Garden https://t.co/OFiK5yf0JQ
    ------------------------------
    nytimes 80
    An R-rated movie with Melissa McCarthy and a cast of puppets can use the tagline "No Sesame. All Street." For now.  https://t.co/bVG1KqEG1y
    ------------------------------
    nytimes 81
    In this environment, free of goals, free of the promise of a new challenge, you become a true human animal https://t.co/wU2D3oLMi8
    ------------------------------
    nytimes 82
    The good, the bad and the ugly of college admissions in the U.S.: college-bound high school seniors tell all https://t.co/Jt7aEuI9Jm
    ------------------------------
    nytimes 83
    A reader who always wondered how her father survived the Holocaust found the answer in one of our obituaries. “I ha… https://t.co/jdOwtN3UPV
    ------------------------------
    nytimes 84
    "What is the responsibility of a sibling for a sibling?" https://t.co/yi7RoKC4a5
    ------------------------------
    nytimes 85
    Egalitarian and sincere, Norway’s version of New Nordic cooking is frisky, witty and unpretentious https://t.co/qxsp1OZtaH
    ------------------------------
    nytimes 86
    Jonathan Kasdan on the romantic comedy that provided entree to "Raiders," the highs and lows of his "Star Wars" wor… https://t.co/az9BXVlhSV
    ------------------------------
    nytimes 87
    Summer, once the dumping ground of television, is now the season with a little something for everyone https://t.co/zah4CJYrIz
    ------------------------------
    nytimes 88
    Bill Clinton: "I wanted to become a politician because I was fascinated by people, policy and politics. I read book… https://t.co/FibiADvaTo
    ------------------------------
    nytimes 89
    Louder background music makes people choose less healthful foods https://t.co/8OBxjOMHUm
    ------------------------------
    nytimes 90
    A photographer's 3-decade exploration of Hanoi, Vietnam, charts the city’s architectural, cultural and economic evo… https://t.co/Svhe9pIiFh
    ------------------------------
    nytimes 91
    In @NYTOpinion 
    
    Op-Ed Contributor Margaret Renkl writes, "I don't know what these legislators are thinking. Maybe… https://t.co/4lTN9bSdvG
    ------------------------------
    nytimes 92
    "I want my son to feel the immediate ease I do when I walk into a room full of Southern voices, but I don’t want ot… https://t.co/nYdLWa8F2z
    ------------------------------
    nytimes 93
    Young adults are not only marrying and having children later in life than previous generations, but taking more tim… https://t.co/qeTRTtsRfg
    ------------------------------
    nytimes 94
    "The thought of either of my 2 sons harassing or assaulting another person, or being victims themselves, is enough… https://t.co/jD5QFTmW10
    ------------------------------
    nytimes 95
    While Berlin certainly features a fair number of luxury options, it hasn’t lost its populist edge https://t.co/yfH4SdbwHo
    ------------------------------
    nytimes 96
    Trump to Hold Ramadan Dinner After Skipping It Last Year https://t.co/tlcCEPZe1i
    ------------------------------
    nytimes 97
    The Ethicist columnist on whether you should inform people if you're carrying an illegal medical drug https://t.co/y0xlqyCaBj
    ------------------------------
    nytimes 98
    A luxurious summer risotto https://t.co/zDkMqBCg5q https://t.co/BllSx4GKxY
    ------------------------------
    nytimes 99
    An ideal vehicle for the ripest strawberries at the height of the season https://t.co/fgar5lOIQT https://t.co/09IxVHN3Ma
    ------------------------------
    


```python
# CREATE DATAFRAME FROM LISTS TO EXPORT AS CSV

df_tweets = pd.DataFrame({"Account":org_list,
                          "Text":text_list,
                          "Date":date_list,
                          "Compound Score":compound_list,
                          "Positive Score":positive_list,
                          "Negative Score":negative_list,
                          "Neutral Score":neutral_list
                         })

# Convert DATE column to datetime and format
df_tweets['Date'] = pd.to_datetime(df_tweets['Date']).dt.strftime('%m/%d/%Y')

df_tweets = df_tweets[['Account',
                       'Date',
                       'Text',
                       'Compound Score',
                       'Positive Score',
                       'Negative Score',
                       'Neutral Score'
                      ]]

df_tweets.to_csv('News Tweets.csv')
```


```python
# CREATE DATAFRAME FOR SCATTER PLOT

df_scatter = pd.DataFrame({"Polarity":compound_list,
                   "Tweets Ago": counter_list,
                   "News Org": org_list})

# Add colors column to the DataFrame for use when plotting
def set_colors (row):
    if row['News Org'] == 'BBCBreaking':
        return 'lightblue'
    if row['News Org'] == 'CBSNews':
        return 'green'
    if row['News Org'] == 'CNN':
        return 'red'
    if row['News Org'] == 'FoxNews':
        return 'blue'
    if row['News Org'] == 'nytimes':
        return 'yellow'
df_scatter['Color'] = df_scatter.apply(set_colors, axis=1)
```


```python
# CREATE SCATTER PLOT SHOWING EVERY TWEET'S POLARITY SCORE

plt.scatter(df_scatter['Tweets Ago'],
            df_scatter['Polarity'],
            alpha=0.7, 
            c = df_scatter['Color'], 
            edgecolors='black',
            zorder = 2)

plt.title('Sentiment Analysis of Media Tweets (' + str(datetime.datetime.now().strftime('%m/%d/%Y')) + ')')
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')

plt.grid(color='white',zorder=1,axis='both')
ax = plt.gca()
ax.set_facecolor('xkcd:light grey')

major_y_ticks = np.arange(-1,1.1,0.5)
ax.set_yticks(major_y_ticks, minor=False)
plt.xlim(105,-5)
plt.ylim(-1.05,1.05)

line1 = lines.Line2D(range(1),range(1),linewidth=0,marker='o',markerfacecolor='lightblue',markeredgecolor='black',label='BBC')
line2 = lines.Line2D(range(1),range(1),linewidth=0,marker='o',markerfacecolor='green',markeredgecolor='black',label='CBS')
line3 = lines.Line2D(range(1),range(1),linewidth=0,marker='o',markerfacecolor='red',markeredgecolor='black',label='CNN')
line4 = lines.Line2D(range(1),range(1),linewidth=0,marker='o',markerfacecolor='blue',markeredgecolor='black',label='Fox News')
line5 = lines.Line2D(range(1),range(1),linewidth=0,marker='o',markerfacecolor='yellow',markeredgecolor='black',label='New York Times')
plt.legend(handles=[line1,line2,line3,line4,line5], 
           title='Media Sources', 
           frameon=False,
           bbox_to_anchor=(1, 1.05))

plt.savefig("Plot_Scatter.png",bbox_inches='tight',pad_inches=0.3)

plt.show()
```


![png](output_5_0.png)



```python
# CREATE BAR CHART

df_bar = pd.DataFrame(sentiment_list)

plt.bar(['BBC','CBS','CNN','Fox','NYT'],
        df_bar['Compound'],
        color=['lightblue','green','red','blue','yellow'],
        width=1,
        edgecolor='black'
       )
plt.ylim(-.25,0)
plt.title('Overall Media Sentiment Based on Twitter (' + str(datetime.datetime.now().strftime('%m/%d/%Y')) + ')')
plt.ylabel('Tweet Polarity')
ax = plt.gca()
ax.set_facecolor('xkcd:light grey')

# For each bar: Place a label 
# THANKS TO justfortherec AT https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart/28931750
rects = ax.patches
for rect in rects:
    # Get X and Y placement of label from rect.
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2

    # Number of points between bar and label
    space = 5
    
    # Vertical alignment for positive values
    va = 'bottom'
    # If value of bar is negative: Place label below bar
    if y_value < 0:
        space *= -1
        va = 'top'

    # Use Y value as label and format number with two decimal places
    label = "{:.2f}".format(y_value)

    # Create annotation
    plt.annotate(
        label,                      
        (x_value, y_value),         
        xytext=(0, space),          
        textcoords="offset points", 
        ha='center',                
        va=va)                     

plt.savefig("Plot_Bar.png",bbox_inches='tight',pad_inches=0.25)
    
plt.show()
```


![png](output_6_0.png)

