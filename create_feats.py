import pandas as pd
import numpy as np


def add_lookback_columns(df, lookback):
    # Check if quality flag column, i.e., enough matches found to compute lookback results
    # exists..  if not, create it
    if 'LB_FLAG' not in df.columns:
        df['LB_FLAG'] = 1

    #Add columns to df and compute recent form for a given number of lookback games
    df["_".join(["LB", str(lookback), "HGF"])] = 0 #Lookback Home goals for
    df["_".join(["LB", str(lookback), "AGF"])] = 0 #Lookback Away goals for
    df["_".join(["LB", str(lookback), "HGA"])] = 0 #Lookback Home goals against
    df["_".join(["LB", str(lookback), "AGA"])] = 0 #Lookback Away goals against
    df["_".join(["LB", str(lookback), "HP"])] = 0 #Lookback Home points
    df["_".join(["LB", str(lookback), "AP"])] = 0 #Lookback Away points



def add_form(df, lookback):

    add_lookback_columns(df, lookback)

    hpoints = {'H': 3, 'D': 1, 'A': 0}
    apoints = {'H': 0, 'D': 1, 'A': 3}

    for index, row in df.iterrows():
        hometeam = row['HomeTeam']
        awayteam = row['AwayTeam']

        #HomeTeam
        lb_hgames = 0  #number of previous games found for home team
        lb_hgf = 0  #cumulative number of goals scored
        lb_hga = 0  #cumulative number of goals conceded
        lb_hpoints = 0  #cumulative number of points
        for j in reversed(xrange(index)):
            if(df.loc[j,'HomeTeam']==hometeam):
                lb_hgf = lb_hgf + df.loc[j,'FTHG']
                lb_hga = lb_hga + df.loc[j,'FTAG']
                lb_hpoints = lb_hpoints + hpoints[df.loc[j,'FTR']]
                lb_hgames+=1
            elif(df.loc[j,'AwayTeam']==hometeam):
                lb_hgf = lb_hgf + df.loc[j,'FTAG']
                lb_hga = lb_hga + df.loc[j,'FTHG']
                lb_hpoints = lb_hpoints + apoints[df.loc[j,'FTR']]
                lb_hgames+=1
            if(lb_hgames==lookback):
                break
        df.loc[index, "_".join(["LB", str(lookback), "HGF"])] = lb_hgf
        df.loc[index, "_".join(["LB", str(lookback), "HGA"])] = lb_hga
        df.loc[index, "_".join(["LB", str(lookback), "HP"])] = lb_hpoints

        #AwayTeam
        lb_agames, lb_agf, lb_aga, lb_apoints = 0, 0, 0, 0  #reinitialize lookback vars
        for j in reversed(xrange(index)):
            if(df.loc[j,'HomeTeam']==awayteam):
                lb_agf = lb_agf + df.loc[j,'FTHG']
                lb_aga = lb_aga + df.loc[j,'FTAG']
                lb_apoints = lb_apoints + hpoints[df.loc[j,'FTR']]
                lb_agames+=1
            elif(df.loc[j,'AwayTeam']==awayteam):
                lb_agf = lb_agf + df.loc[j,'FTAG']
                lb_aga = lb_aga + df.loc[j,'FTHG']
                lb_apoints = lb_apoints + apoints[df.loc[j,'FTR']]
                lb_agames+=1
            if(lb_agames==lookback):
                break
        df.loc[index, "_".join(["LB", str(lookback), "AGF"])] = lb_agf
        df.loc[index, "_".join(["LB", str(lookback), "AGA"])] = lb_aga
        df.loc[index, "_".join(["LB", str(lookback), "AP"])] = lb_apoints

        if (lb_hgames<lookback) | (lb_agames<lookback):
            df.loc[index,'LB_FLAG'] = 0
        #else:
        #    df.loc[index,'LB_FLAG'] = 0
