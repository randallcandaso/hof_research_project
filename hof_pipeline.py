"""

Previous research has applied different machine learning algorithms to help analyze Hall of Fame voting patterns in professional baseball, 
often with the goal of identifying which statistics voters have prioritized in the past. However, few studies have translated this analysis 
into developing further predictive tools for evaluating current player performance. This study aims to create a predictive modeling pipeline 
that not only understands the statistical values of induction, but utilizes that understanding to predict which currently active Major League 
Baseball players are on a Hall of Fame trajectory. The data collected for this research included the hitting and baserunning statistics of over 
300 historical and 250 active MLB players. The pipeline incorporates a variety of modeling techniques including soft clustering to identify 
patterns in seasonal performance, L1-regularization for feature selection, XGBoost regression to estimate remaining career production, and Random 
Forest classification to predict Hall of Fame status. After extensive processing and modeling, the pipeline concluded that out of the currently 
active player pool, 23 modern players were identified to be on the path to Hall of Fame induction. The objective of this modeling pipeline is to 
demonstrate and support the usage of machine learning techniques in the decision-making process of evaluating long-term outcomes in the sporting world.

"""


from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from imblearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import sqlite3
import os


def main():
    df_list = load_dfs()
    
    new_active, new_hof, new_retired = filter_dfs(df_list[1], df_list[0], df_list[2])
    
    filtered_active, filtered_hof, filtered_retired = filter_dfs(new_active, new_hof, new_retired)
    
    inactive_data = pd.merge(filtered_hof, filtered_retired, how='outer')
    
    active_data = setup_awards(filtered_active, filtered_active, filtered_hof, filtered_retired, df_list[4], df_list[6],
                 df_list[5], df_list[7], df_list[9], df_list[8], df_list[3], check=True)
    inactive_data = setup_awards(inactive_data, filtered_active, filtered_hof, filtered_retired, df_list[4], df_list[6],
                 df_list[5], df_list[7], df_list[9], df_list[8], df_list[3])
    
    create_status(inactive_data, filtered_hof)

    inactive_data, active_data, clusters_inactive, cluster_active, final_inactive, final_active = preprocess_helper(inactive_data, active_data)
    
    print('------------------------')
    final_inactive, final_active = feature_reduction(final_inactive, final_active)
    print('------------------------')
    active_projections, final_inactive = projection_helper(inactive_data, clusters_inactive, final_active, final_inactive)
    print('------------------------')
    hof_pipeline = classification_preds(final_inactive)
    print('------------------------')
    final_players = get_active_hofs(active_projections, hof_pipeline)
    print('Listed Below Are Active Players With HOF Trajectory:')s
    print(final_players)


def load_dfs():
    '''
    
    The needed datasets for the program are loaded in and properly formatted for future use.

    return: needed datasets (list)
    
    '''
    pd.options.mode.chained_assignment = None
    
    col_order = ['Player', 'Team', 'Season', 'Age', 'G', 'PA', 'AB',
           'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'XBH', 'SB', 'CS', 'BB', 'SO', 'BA',
           'OBP', 'SLG', 'OPS', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
           'WAR', 'Pos']
    
    players_hof = pd.read_csv('Hall of Fame by Seasons.csv', encoding='utf-8')
    players_hof = players_hof.replace({'Â': '', 'Ã©': 'é', 'Ã': '', '±': 'ñ', '©': 'é', '':'Á', 'º': 'ú'}, regex=True)
    players_hof = players_hof.drop(players_hof.columns[0], axis=1)
    players_hof = players_hof.iloc[:-2]
    players_hof = players_hof[col_order]
    hof_total = pd.read_csv('Hall of Fame.csv', encoding='utf-8')
    
    players_retired = pd.read_csv('Retired Players by Seasons.csv', encoding='utf-8', index_col=0)
    players_retired.index.name = None
    players_retired = players_retired.replace({'Â': '', 'Ã©': 'é', 'Ã': '', '±': 'ñ', '':'Á', 'º': 'ú', '©': 'é'}, regex=True)
    players_retired = players_retired[col_order]
    retired_total = pd.read_csv('Retired.csv', encoding='utf-8')
    
    players_active = pd.read_csv('Active Player Seasons.csv', encoding='utf-8', index_col=0)
    players_active.index.name = None
    players_active = players_active.replace({'Â': '', 'Ã©': 'é', 'Ã': '', '±': 'ñ', '©': 'é', '':'Á', 'º': 'ú'}, regex=True)
    players_active = players_active[col_order]
    
    temp_stars = pd.read_csv('Full All Stars by Season.csv', encoding='ISO-8859-1')
    all_stars = temp_stars.iloc[:, :4].copy()
    all_stars.columns = ['Year', 'First', 'Last', 'Position']
    all_stars = all_stars.replace({'Â': '', 'Ã©': 'é', 'Ã': '', '±': 'ñ'}, regex=True)
    all_stars['Player'] = all_stars['First'] + ' ' + all_stars['Last']
    all_stars['Player'] = (all_stars['First'] + ' ' + all_stars['Last']).str.strip()
    all_stars = all_stars.drop(['First', 'Last'], axis=1)
    
    al_gg = pd.read_csv('AL Gold Glovers.csv', encoding='utf-8')
    al_ss = pd.read_csv('AL Silver Sluggers.csv', encoding='utf-8')
    nl_gg = pd.read_csv('NL Gold Glovers.csv', encoding='utf-8')
    nl_ss = pd.read_csv('NL Silver Sluggers.csv', encoding='utf-8')
    bat_titles = pd.read_csv('Batting Titles.csv', encoding='utf-8')
    roy_winners = pd.read_csv('ROY.csv', encoding='utf-8', header=1)
    hank_aarons = pd.read_csv('Hank Aaron.csv', encoding='utf-8')
    mvp_winners = pd.read_csv('MVP.csv', encoding='utf-8')

    return [players_hof,players_active,players_retired,all_stars,al_gg,al_ss,nl_gg,nl_ss,bat_titles,mvp_winners]


def fix_pos(df):
    '''
    
    df: dataframe to be formatted (pd.DataFrame)

    The position column values are edited for just the primary position of a player.
    
    '''
    for i in df.index:
        positions = df.loc[i, 'Pos']
        for char in positions:
            if char.isdigit() or char == 'D':
                if char == 'D':
                    char = '10'
                df.loc[i, 'New_Pos'] = char
                break
                

def filter_dfs(players_active, players_hof, players_retired):
    '''
    
    players_active: active playerbase (pd.DataFrame)
    players_hof: hof playerbase (pd.DataFrame)
    players_retired: retired playerbae (pd.DataFrame)

    The inputted dataframes are filtered to reduce sample size of each population
    as well as to make sure each player's data is eligible for use.

    return: filtered dataframes (pd.DataFrame)

    '''

    new_active = players_active[players_active['Season'] != 2025]
    
    new_active = new_active.fillna(0)
    
    conn = sqlite3.connect(':memory:')
    new_active.to_sql('new_active', conn, index=False, if_exists='replace')
    
    query = """
    SELECT *
    FROM new_active
    WHERE Player IN (
        SELECT Player
        FROM new_active
        GROUP BY Player
        HAVING COUNT(*) >= 4
    )
    """
    
    new_active = pd.read_sql_query(query, conn)
    
    new_hof = players_hof.fillna(0)
    
    conn = sqlite3.connect(':memory:')
    new_hof.to_sql('new_hof', conn, index=False, if_exists='replace')
    
    query = """
    SELECT *
    FROM new_hof
    WHERE Player IN (
        SELECT Player
        FROM new_hof
        WHERE Season >= 1950
        GROUP BY Player
        HAVING COUNT(*) >= 10
    )
    """
    
    new_hof = pd.read_sql_query(query, conn)
    
    new_retired = players_retired.fillna(0)
    
    conn = sqlite3.connect(':memory:')
    new_retired.to_sql('new_retired', conn, index=False, if_exists='replace')
    
    query = """
    SELECT *
    FROM new_retired
    WHERE Player IN (
        SELECT Player
        FROM new_retired
        WHERE Season >= 1950
        GROUP BY Player
        HAVING COUNT(*) >= 10
    )
    """
    
    new_retired = pd.read_sql_query(query, conn)

    new_retired['Career_WAR'] = new_retired.groupby('Player')['WAR'].transform('sum')

    new_retired = new_retired[new_retired['Career_WAR'] >= 30]

    new_retired = new_retired.drop(['Career_WAR'], axis=1)

    fix_pos(new_active)
    fix_pos(new_hof)
    fix_pos(new_retired)

    return new_active, new_hof, new_retired


def check_active(act_temp, year, glover):
    '''

    act_temp: active dataframe (pd.DataFrame)
    year: year of season (int)
    glover: player info (list)

    Checks the active dataframe if the given player is currently active.

    return: player name (str), year of season (int)

    '''
    for i in act_temp.index:
        player = act_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and int(player['Season']) == int(year):
            if player['Team'] == glover[1]:
                return player['Player'], int(year)
    return None, None


def check_hof(hof_temp, year, glover):
    '''
    
    hof_temp: hof dataframe (pd.DataFrame)
    year: year of season (int)
    glover: player info (list)

    Checks the HOF dataframe to check if the given player is a Hall of Famer.

    return: player name (str), year of season (int)

    '''
    for i in hof_temp.index:
        player = hof_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and int(player['Season']) == int(year):
            if player['Team'] == glover[1]:
                return player['Player'], int(year)
    return None, None


def check_retired(ret_temp, year, glover):
    '''
    
    ret_temp: retired dataframe (pd.DataFrame)
    year: year of season (int)
    glover: player info (list)

    Checks the retired dataframe to check if the given player is retired.

    return: player name (str), year of season (int)
    
    '''
    for i in ret_temp.index:
        player = ret_temp.loc[i]
        name = player['Player'].split()
        if name[1] == glover[0] and int(player['Season']) == int(year):
            if player['Team'] == glover[1]:
                return player['Player'], int(year)
    return None, None


def setup_temps(df):
    '''
    
    df: temporary dataframe (pd.DataFrame)

    Helper function that reformats the given dataframe.
    
    return: reformatted dataframe (pd.DataFrame)
    
    '''
    df = df.copy()
    df.set_index(df.columns[0], inplace=True)
    team = 'Team'
    if team in df.columns:
        df.drop('Team', axis=1, inplace=True)
    return df


def gg_ss_checks(league, whole, col, active, hof, retired):
    '''
    
    league: award dataframe specific to league (pd.DataFrame)
    whole: dataframe to be counted for (pd.DataFrame)
    col: award to be counted (str)
    active: active playerbase (pd.DataFrame)
    hof: hof playerbase (pd.DataFrame)
    retired: retired playerbase (pd.DataFrame)

    Given an award and an inputted dataframe, the function checks the status of each player and 
    accounts for if they won the inputted award in a specific season.
    
    '''
    for i in league.index:
        for j in league.columns:
            if pd.isna(league.loc[i, j]):
                continue
            glover = league.loc[i,j].split('·')
            glover = [item.strip('\xa0') for item in glover]
            year = i.split()
            check_1, season_1 = check_active(active, year[0], glover)
            check_2, season_2 = check_hof(hof, year[0], glover)
            check_3, season_3 = check_retired(retired, year[0], glover)
            if check_1:
                whole.loc[(whole['Player'] == check_1) & (whole['Season'] == season_1), col] = 1
            if check_2:
                whole.loc[(whole['Player'] == check_2) & (whole['Season'] == season_2), col] = 1
            if check_3:
                whole.loc[(whole['Player'] == check_3) & (whole['Season'] == season_3), col] = 1


def simple_awards(df, award_df, award):
    '''
    
    df: dataframe to be counted for (pd.DataFrame)
    award_df: award dataframe (pd.DataFrame)
    award: award to be counted (str)

    Given an award and an inputted dataframe, the function accounts for if they won the inputted 
    award in a specific season.
    
    '''
    df[award] = 0
    award_df_2 = award_df.copy()

    for i in award_df_2.index:
        player = award_df_2.iloc[i, 2]
        season = award_df_2.iloc[i, 0]
        for j in df.index:
            temp_1 = df.loc[j, 'Player']
            temp_2 = df.loc[j, 'Season']
            if temp_1 == player:
                if temp_2 == season:
                        df.loc[j, award] = 1

                    
def setup_awards(df, active, hof, retired,
                 al_gg, nl_gg, al_ss, nl_ss, mvp_winners,
                 bat_titles, all_stars, check=False):
    '''

    df: dataframe to be counted for (pd.DataFrame)
    active: active playerbase (pd.DataFrame)
    hof: hof playerbase (pd.DataFrame)
    retired: retired playerbase (pd.DataFrame)
    al_gg: AL gold gloves (pd.DataFrame)
    nl_gg: NL gold gloves (pd.DataFrame)
    al_ss: AL silver sluggers (pd.DataFrame)
    nl_ss: NL silver sluggers (pd.DataFrame)
    mvp_winners: MVP winners (pd.DataFrame)
    bat_titles: batting title winners (pd.DataFrame)
    check: checks for active status (Boolean)
    
    Helper function that helps setup the award dataframes and input values for the award-count functions.

    return: updated dataframe with award counts (pd.DataFrame)
    
    '''
    
    if check:
        df = df[df['Season'] != 2025]
        df['status'] = 'active'

    new_df = df.copy()

    new_df['GGs'] = 0
    al_gg_temp = setup_temps(al_gg)
    nl_gg_temp = setup_temps(nl_gg)
    
    gg_ss_checks(al_gg_temp, new_df, 'GGs', active, hof, retired)
    gg_ss_checks(nl_gg_temp, new_df, 'GGs', active, hof, retired)
    
    new_df['SSs'] = 0
    al_ss_temp = setup_temps(al_ss)
    nl_ss_temp = setup_temps(nl_ss)
    
    gg_ss_checks(al_ss_temp, new_df, 'SSs', active, hof, retired)
    gg_ss_checks(nl_ss_temp, new_df, 'SSs', active, hof, retired)
    
    simple_awards(new_df, mvp_winners, 'MVPs')
    
    bt_2 = bat_titles.copy()
    bt_2['Batting Champ'] = bt_2['Batting Champ'].str.replace('\xa0', ' ')
    simple_awards(new_df, bt_2, 'Bat_Titles')
    
    as_2 = all_stars.copy()
    simple_awards(new_df, as_2, 'All_Stars')

    return new_df


def create_status(inactive_data, players_hof):
    '''
    
    inactive_data: total inactive playerbase (pd.DataFrame)
    players_hof: hof playerbase (pd.DataFrame)

    Each inactive player is checked for HOF status and updates the 'status' column if indeed
    a member.
    
    '''
    inactive_data["status"] = 'retired'
    
    hof_check = []
    for i in players_hof.index:
        star = players_hof.loc[i,'Player']
        if star not in hof_check:
            hof_check.append(star)
    
    for i in inactive_data.index:
        player = inactive_data.at[i,'Player']
        if player in hof_check:
            inactive_data.loc[i,'status'] = 'hof'

def fix_seasons(df):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)

    The season column is edited to display the number of seasons active up to that point of a player's
    career.
    
    '''
    for player in df['Player'].unique():
        debut = df.loc[df['Player'] == player, 'Season'].min()
        df.loc[df['Player'] == player, 'Season'] = df.loc[df['Player'] == player, 'Season'] - debut + 1


def new_cols(inactive_data, active_data):
    '''
    
    inactive_data: inactive playerbase (pd.DataFrame)
    active_data: active playerbase (pd.DataFrame)

    Both inputted dataframes' columns are reordered to match the proper column order needed for future use.

    return: reformatted dataframes (pd.DataFrame)
    
    '''
    col_order = ['Player', 'Team', 'Season', 'Age', 'G', 'PA', 'AB',
           'R', 'H', '1B', '2B', '3B', 'HR', 'RBI', 'XBH', 'SB', 'CS', 'BB', 'SO', 'BA',
           'OBP', 'SLG', 'OPS', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB',
           'WAR', 'GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs', 'Pos', 'status']
    
    inactive_data = inactive_data[col_order]
    active_data = active_data[col_order]
    return inactive_data, active_data


def dummy_pos(df):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)

    Dummy variables are created and added to the original dataframe for the position column. 

    return: dataframe with dummy variables (pd.DataFrame)
    
    '''
    temp = df.copy()
    temp_dummies = pd.get_dummies(temp['Pos'], prefix='Pos')
    temp_status = df['status']
    temp.drop(['Pos', 'status'], axis=1, inplace=True)
    new_df = pd.concat([temp, temp_dummies, temp_status], axis=1)
    
    new_df = new_df[new_df['Pos_1'] != True]
    new_df.drop(['Pos_1'], axis=1, inplace=True)
    return new_df


def fix_skew(df, col, min_abs_skew=0.5):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)
    col: column to be transformed (str)
    min_abs_skew: minimum absolute value (float)

    Given a column's distribution in a dataframe, the column is mathematically transformed so it is 
    normally distributed.
    
    '''
    skew = df[col].skew()
    abs_skew = abs(skew)
    
    if abs_skew <= min_abs_skew:
        return
    
    if skew < 0:
        max_val = df[col].max()
        df[col] = max_val - df[col] + 1 
        
    min_val = df[col].min()
    if min_val <= 0:
        offset = abs(min_val) + 1e-6
        df[col] = df[col] + offset
    
    if min_abs_skew <= abs_skew < 1.0:
        df[col] = np.sqrt(df[col])
    elif abs_skew >= 1.0:
        df[col] = np.log1p(df[col]) 


def fix_pos_col(df):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)

    Helper function that renames the new position column.

    return: adjusted dataframe (pd.DataFrame)
    
    '''
    df = df.drop(columns='Pos').rename(columns={'New_Pos': 'Pos'})
    return df


def find_optimal_clusters(data, max_clusters=8):
    '''
    
    data: numeric data (pd.DataFrame)
    max_clusters: maximum desired clusters (int)

    The silhoutte score of each cluster number is calculated to determine the optimal number of 
    clusters for this data. 

    return: optimal cluster count (int)
    
    '''
    scores = []
    print('Clustering Performance:')
    for k in range(2, max_clusters+1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        clusters = gmm.fit_predict(data)
        score = silhouette_score(data, clusters)
        scores.append(score)
        print(f"Clusters: {k} - Silhouette: {score:.3f}")
    
    optimal_k = np.argmax(scores) + 2
    print(f"Optimal cluster count: {optimal_k}")
    return optimal_k


def calc_bat_avg(hits, at_bats):
    '''
    
    hits: number of hits (int)
    at_bats: number of at-bats (int)

    A player's batting average is calculated.

    return: batting percentage (float)
    
    '''
    if at_bats == 0:
        return 0.0
    return round((hits/at_bats), 3)


def calc_obp(hits, walks, hbp, at_bats, sf):
    '''
    
    hits: number of hits (int)
    walks: number of walks (int)
    hbp: number of hit-by-pitches (int)
    at_bats: number of at-bats (int)
    sf: number of sacrificial flies (int)

    A player's on-base percentage is calculated.

    return: on-base percentage (float)
    
    '''
    first = hits + walks + hbp
    second = at_bats + hbp + sf
    if second == 0:
        return 0.0
    return round((first/second), 3)


def calc_slug(singles, doubles, triples, hrs, at_bats):
    '''
    
    singles: number of first-base hits (int)
    doubles: number of double-base hits (int)
    triples: number of triple-base hits (int)
    hrs: number of home runs (int)
    at_bats: number of at-bats (int)

    A player's slugging percentage is calculated.

    return: slugging percentage (float)
    
    '''
    if at_bats == 0:
        return 0.0
    doubles = doubles * 2
    triples = triples * 3
    home_bs = hrs * 4
    return round(((singles+doubles+triples+home_bs)/at_bats), 3)


def calc_ops(obp, slug):
    '''
    
    obp: on-base percentage (float)
    slug: slugging percentage (float)

    A player's OPS value is calculated.

    return: OPS value (float)
    
    '''
    return round((obp+slug), 3)


def total_df_helper(df):
    '''
    
    df: seasonal dataframe (pd.DataFrame)

    Taking the seasonal performance data of each player, their respective career totals are
    calculated and inputted into a new dataframe.

    return: career totals dataframe (pd.DataFrame)

    '''
    col_names = ['Player',  'Age', 'Season_num', 'G', 'PA', 'AB', 'R', 'H', '1B',
       '2B', '3B', 'HR', 'RBI', 'XBH', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP',
       'SLG', 'OPS', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB', 'WAR',
       'Pos_2', 'Pos_3', 'Pos_4', 'Pos_5', 'Pos_6', 'Pos_7', 'Pos_8', 'Pos_9', 'Pos_10',
       'GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs', 'Cluster_0_Prob', 
        'Cluster_1_Prob', 'status']

    empty = []
    for player in df['Player'].unique():
        player_df = df[df['Player'] == player]

        hits = player_df['H'].sum()
        at_bats = player_df['AB'].sum()
        walks = player_df['BB'].sum()
        hbp = player_df['HBP'].sum()
        sf = player_df['SF'].sum()
        singles = player_df['1B'].sum()
        doubles = player_df['2B'].sum()
        triples = player_df['3B'].sum()
        hrs = player_df['HR'].sum()

        obp = calc_obp(hits, walks, hbp, at_bats, sf)
        slg = calc_slug(singles, doubles, triples, hrs, at_bats)

        total_pa = player_df['PA'].sum()
        avg_cluster0 = np.average(player_df['Cluster_0_Prob'], weights=player_df['PA'])
        avg_cluster1 = np.average(player_df['Cluster_1_Prob'], weights=player_df['PA'])

        temp = [player, player_df['Age'].max(), player_df['Season'].nunique(), player_df['G'].sum(),
                total_pa, at_bats, player_df['R'].sum(), hits, singles, doubles, triples, hrs,
                player_df['RBI'].sum(), player_df['XBH'].sum(), player_df['SB'].sum(), player_df['CS'].sum(),
                walks, player_df['SO'].sum(), calc_bat_avg(hits, at_bats), obp, slg, calc_ops(obp, slg),
                player_df['TB'].sum(), player_df['GIDP'].sum(), hbp, player_df['SH'].sum(), sf, 
                player_df['IBB'].sum(), player_df['WAR'].sum(), 
                player_df['Pos_2'].values[0], player_df['Pos_3'].values[0], player_df['Pos_4'].values[0],
                player_df['Pos_5'].values[0], player_df['Pos_6'].values[0], player_df['Pos_7'].values[0],
                player_df['Pos_8'].values[0], player_df['Pos_9'].values[0], player_df['Pos_10'].values[0],
                player_df['GGs'].sum(), player_df['SSs'].sum(), player_df['Bat_Titles'].sum(),
                player_df['All_Stars'].sum(), player_df['MVPs'].sum(), avg_cluster0, avg_cluster1, 
                player_df['status'].unique()[0]]
        
        empty.append(temp)

    new_df = pd.DataFrame(data=empty, columns=col_names)
    return new_df


def calc_clust_probs(inactive_data, active_data):
    '''
    
    inactive_data: inactive playerbase (pd.DataFrame)
    active_data: active playerbase (pd.DataFrame)

    Cluster probabilities are created based on the inactive data and then applied to the active
    playerbase. 

    return: new dataframes with cluster values (pd.DataFrame)
    
    '''
    os.environ['LOKY_MAX_CPU_COUNT'] = '7'
    
    test_inactive = inactive_data.copy()
    test_inactive.drop(['GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs'], axis=1, inplace=True)
    
    test_active = active_data.copy()
    test_active.drop(['GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs'], axis=1, inplace=True)
    
    test_inactive_X = test_inactive[test_inactive.columns[2:39]].dropna()
    test_active_X = test_active[test_active.columns[2:39]].dropna()
    
    for i in test_inactive_X.columns[2:30]:
        fix_skew(test_inactive_X, i)
    
    for i in test_active_X.columns[2:30]:
        fix_skew(test_active_X, i)
    
    pca = PCA(n_components=11)
    inactive_X_pca = pca.fit_transform(test_inactive_X)
    active_X_pca = pca.transform(test_active_X)
    
    best_k = find_optimal_clusters(inactive_X_pca)
    
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm.fit(inactive_X_pca)
    
    cluster_probs_inactive = gmm.predict_proba(inactive_X_pca)
    cluster_probs_active = gmm.predict_proba(active_X_pca)

    for i in range(cluster_probs_inactive.shape[1]):
        test_inactive[f'Cluster_{i}_Prob'] = cluster_probs_inactive[:, i]

    for i in range(cluster_probs_active.shape[1]):
        test_active[f'Cluster_{i}_Prob'] = cluster_probs_active[:, i]

    return test_inactive, test_active


def total_dfs(inactive_data, active_data, test_inactive, test_active):
    '''
    
    inactive_data: seasonal inactive playerbase (pd.DataFrame)
    active_data: seasonal active playerbase (pd.DataFrame)
    test_inactive: inactive playerbase with cluster values (pd.DataFrame)
    test_active: active playerbase with cluster values (pd.DataFrame)

    The seasonal data with its matching cluster values are reformatted and calculated for 
    their career total values in newly created dataframes.
    
    return: new career total dataframes (pd.DataFrame)
    
    '''
    awards = ['GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs']
    awards_inactive = inactive_data[awards]
    
    final_inactive = pd.concat([test_inactive.copy(), awards_inactive], axis=1)
    final_inactive = total_df_helper(final_inactive)
    
    awards = ['GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs']
    awards_active = active_data[awards]
    
    final_active = pd.concat([test_active.copy(), awards_active], axis=1)
    final_active = total_df_helper(final_active)
    return final_inactive, final_active


def preprocess_helper(inactive, active):
    '''
    
    inactive: inactive playerbase (pd.DataFrame)
    active: active playerbase (pd.DataFrame)

    The two inputted dataframes are preprocessed and then used for caclulation of their cluster
    probability values. Taking their seasonal performance data and its matching cluster probabilities, 
    career total dataframes are then created. 

    return: different use-case dataframes respective to status (pd.DataFrame)
    
    '''
    fix_seasons(inactive)
    fix_seasons(active)

    inactive = fix_pos_col(inactive)
    active = fix_pos_col(active)
    
    inactive_data, active_data = new_cols(inactive, active)
    
    inactive_data = dummy_pos(inactive_data)
    active_data = dummy_pos(active_data)

    clusters_inactive, cluster_active = calc_clust_probs(inactive_data, active_data)
    final_inactive, final_active = total_dfs(inactive_data, active_data, clusters_inactive, cluster_active)
    
    return inactive_data, active_data, clusters_inactive, cluster_active, final_inactive, final_active


def feature_reduction(inactive, active):
    '''
    
    inactive: inactive playerbase (pd.DataFrame)
    active: active playerbase (pd.DataFrame)

    The feature set is reduced using Lasso Regression with L1 penalization, a method of regularization. It should
    be noted that the feature set is manually adjusted based on knowledge from prior programming runs.

    return: new dataframes with reduced feature sets (pd.DataFrame)
    
    '''
    numeric = ['Age', 'Season_num', 'G', 'PA', 'AB', 'R', 'H', '1B', '2B',
           '3B', 'HR', 'RBI', 'XBH', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG',
           'OPS', 'TB', 'GIDP', 'HBP', 'SH', 'SF', 'IBB', 'WAR', 'Cluster_0_Prob',
           'Cluster_1_Prob'] 
    
    dummies = ['Pos_2', 'Pos_3', 'Pos_4', 'Pos_5', 'Pos_6', 'Pos_7', 'Pos_8', 
               'Pos_9', 'Pos_10', 'GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs']
    
    X = inactive.iloc[:, 1:45]
    scaler = StandardScaler()
    
    X_scaled = scaler.fit_transform(X[numeric])
    X_scaled = pd.DataFrame(X_scaled, columns=numeric, index=X.index)
    X_test = pd.concat([X_scaled, X[dummies]], axis=1)
    y = inactive['status'].map({'hof': 1, 'retired': 0})
    
    logistic_lasso = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', max_iter=10_000).fit(X_test, y)
    
    selected_features = X_test.columns[logistic_lasso.coef_[0] != 0]

    to_remove = ['R', 'XBH', 'GIDP', 'SH', 'CS', 'IBB', 'SLG']
    selected_features = ['2B', '3B', 'HR', 'RBI', 'SB', 'SO', 'HBP', 'SF', 'WAR',
       'Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 'Pos_8', 'Pos_9', 'GGs', 'SSs',
       'All_Stars', 'MVPs', 'Cluster_0_Prob', 'Cluster_1_Prob', 'BA', 'OPS']

    final_features = (['Player', 'Age', 'Season_num', 'PA', 'G'] +
    [i for i in selected_features if i not in to_remove] + ['status'])
    print(f"Feature Set Went From {len(inactive.columns[1:45])} to {len(final_features)}")
    print("Selected features:", final_features[:29])

    final_inactive = inactive[final_features].fillna(0)
    final_active = active[final_features].fillna(0)

    return final_inactive, final_active


def add_rate_stats(df):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)

    Statistical rates are calculated for each listed column. These rates are inputted as new columns.
    
    '''
    df['HR_per_500_PA'] = (df['HR'] / df['PA']) * 500
    df['RBI_per_500_PA'] = (df['RBI'] / df['PA']) * 500
    df['SB_per_500_PA'] = (df['SB'] / df['PA']) * 500
    df['SO_per_500_PA'] = (df['SO'] / df['PA']) * 500
    df['HBP_per_500_PA'] = (df['HBP'] / df['PA']) * 500
    df['SF_per_500_PA'] = (df['SF'] / df['PA']) * 500
    df['WAR_per_500_PA'] = (df['WAR'] / df['PA']) * 500
    df['2B_per_500_PA'] = (df['2B'] / df['PA']) * 500
    df['3B_per_500_PA'] = (df['3B'] / df['PA']) * 500


def safe_weighted_avg(values, weights):
    weights = weights + 1e-6
    valid = ~np.isnan(values)
    if not np.any(valid):
        return np.nan
    return np.average(values[valid], weights=weights[valid])

    
def ages_df(df):
    '''
    
    df: dataframe to be adjusted (pd.DataFrame)

    The seasonal inactive data is utilized to create a new dataframe containing information of each historical player in 
    different stages of their careers. 

    return: career-stage dataframe (pd.DataFrame)
    
    '''
    col_names = ['Player', 'Age', 'G', 'Season_num', 'PA_total', 'PA_left', 'Pos_2','Pos_5', 'Pos_6', 'Pos_7',
                  'Pos_8', 'Pos_9', 'Cluster_0_Prob', 'Cluster_1_Prob', 'BA', 'OPS']

    empty = []
    for player in df['Player'].unique():
        player_df = df[df['Player'] == player]

        total_pa = player_df['PA'].sum()
        career_years = player_df['Season'].max()

        young = player_df[player_df['Season'] <= 4]
        prime = player_df[player_df['Season'] <= 8]
        old = player_df[player_df['Season'] <= 12]

        young_ba = calc_bat_avg(young['H'].sum(), young['AB'].sum())
        young_obp = calc_obp(young['H'].sum(), young['BB'].sum(), young['HBP'].sum(), young['AB'].sum(), young['SF'].sum())
        young_slg = calc_slug(young['1B'].sum(), young['2B'].sum(), young['3B'].sum(), young['HR'].sum(), young['AB'].sum())
        
        young_temp = [player, young['Age'].max(), young['G'].sum(),  young['Season'].nunique(), total_pa,
                      total_pa - young['PA'].sum(), player_df['Pos_2'].values[0], player_df['Pos_5'].values[0],
                      player_df['Pos_6'].values[0], player_df['Pos_7'].values[0], player_df['Pos_8'].values[0],
                      player_df['Pos_9'].values[0], safe_weighted_avg(young['Cluster_0_Prob'].values, young['PA'].values), 
                      safe_weighted_avg(young['Cluster_1_Prob'].values, young['PA'].values), young_ba,
                      calc_ops(young_obp, young_slg)]

        empty.append(young_temp)

        prime_ba = calc_bat_avg(prime['H'].sum(), prime['AB'].sum())
        prime_obp = calc_obp(prime['H'].sum(), prime['BB'].sum(), prime['HBP'].sum(), prime['AB'].sum(), prime['SF'].sum())
        prime_slg = calc_slug(prime['1B'].sum(), prime['2B'].sum(), prime['3B'].sum(), prime['HR'].sum(), prime['AB'].sum())
        
        prime_temp = [player, prime['Age'].max(), prime['G'].sum(), prime['Season'].nunique(), total_pa,
                      total_pa - prime['PA'].sum(), player_df['Pos_2'].values[0], player_df['Pos_5'].values[0],
                      player_df['Pos_6'].values[0], player_df['Pos_7'].values[0], player_df['Pos_8'].values[0],
                      player_df['Pos_9'].values[0], safe_weighted_avg(prime['Cluster_0_Prob'].values, prime['PA'].values), 
                      safe_weighted_avg(prime['Cluster_1_Prob'].values, prime['PA'].values), prime_ba,
                      calc_ops(prime_obp, prime_slg)]

        empty.append(prime_temp)

        old_ba = calc_bat_avg(old['H'].sum(), old['AB'].sum())
        old_obp = calc_obp(old['H'].sum(), old['BB'].sum(), old['HBP'].sum(), old['AB'].sum(), old['SF'].sum())
        old_slg = calc_slug(old['1B'].sum(), old['2B'].sum(), old['3B'].sum(), old['HR'].sum(), old['AB'].sum())
        
        old_temp = [player, old['Age'].max(), old['G'].sum(), old['Season'].nunique(), total_pa, 
                    total_pa - old['PA'].sum(), player_df['Pos_2'].values[0], player_df['Pos_5'].values[0],
                    player_df['Pos_6'].values[0], player_df['Pos_7'].values[0], player_df['Pos_8'].values[0],
                    player_df['Pos_9'].values[0], safe_weighted_avg(old['Cluster_0_Prob'].values, old['PA'].values), 
                    safe_weighted_avg(old['Cluster_1_Prob'].values, old['PA'].values), old_ba, calc_ops(old_obp, old_slg)]

        empty.append(old_temp)

    new_df = pd.DataFrame(data=empty, columns=col_names)
    return new_df


def projection_helper(inactive_data, clusters_inactive, final_active, final_inactive):
    '''
    
    inactive_data: seasonal inactive dataset with awards (pd.DataFrame)
    clusters_inactive: seasonal inactive dataset with cluster probabilities (pd.DataFrame)
    final_active: career total active dataset (pd.DataFrame)
    final_inactive: career total inactive dataset (pd.DataFrame)

    A model is trained based on the career stages of historical players, to predict the number of plate
    appearences left given the stage of career an inputted player is currently at. Statistical rates are
    caclulated for the active player pool. A predicted amount of plate appearences is predicted for each
    active player and along with their matching statistical rates, a projected end of career total for 
    each statistic is summed in a new projection dataframe.

    return: final dataframes to be used for prediction (pd.DataFrame)
    
    '''
    awards = ['GGs', 'SSs', 'Bat_Titles', 'All_Stars', 'MVPs']
    awards_inactive = inactive_data[awards]
    
    phases_inactive = pd.concat([clusters_inactive.copy(), awards_inactive], axis=1)
    phases_inactive = ages_df(phases_inactive)
    
    rates_active = final_active.copy()
    add_rate_stats(rates_active)
    
    rates_order = ['Player', 'PA', 'G', 'Age', 'Season_num', 'Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 
                    'Pos_8', 'Pos_9', 'BA', 'OPS', 'Cluster_0_Prob', 'Cluster_1_Prob', 
                    'HR_per_500_PA', 'RBI_per_500_PA', 'SB_per_500_PA',
                     'SO_per_500_PA', 'HBP_per_500_PA', 'SF_per_500_PA', 'WAR_per_500_PA',
                   '2B_per_500_PA', '3B_per_500_PA', 'status']
    
    rates_active = rates_active[rates_order]

    pa_X = phases_inactive[['Age', 'G', 'Season_num', 'Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 'Pos_8', 'Pos_9',  
                        'Cluster_0_Prob', 'Cluster_1_Prob', 'BA', 'OPS']]

    pa_y = phases_inactive['PA_left']
    
    pa_X_train, pa_X_test, pa_y_train, pa_y_test = train_test_split(pa_X, pa_y, test_size=0.2, random_state=42)
    
    pa_model = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.1, n_jobs=-1)
    
    scores = cross_val_score(pa_model, pa_X_train, pa_y_train, cv=5, scoring='r2', n_jobs=-1)
    print("XGBRegressor Performance:")
    print(f"PA R^2 CV: {scores.mean():.3f} ± {scores.std():.3f}")
    
    pa_model.fit(pa_X_train, pa_y_train)
    test_r2 = pa_model.score(pa_X_test, pa_y_test)
    print(f"Test R^2: {round(test_r2, 3)}")

    return create_projections(pa_model, rates_active, final_active, final_inactive)


def create_projections(model, rates_active, final_active, final_inactive):
    '''
    
    model: remaining plate appearences model (XGBRegressor)
    rates_active: active player statistical rates (pd.DataFrame)
    final_active: career total active dataset (pd.DataFrame)
    final_inactive: career total inactive dataset (pd.DataFrame)

    End of career projections for each target statistic is calculate for using the inputted
    model and the statistical rates for a given player. 

    return: final dataframes to be used for prediction (pd.DataFrame)
    
    '''
    targets = ['2B', '3B', 'HR', 'RBI', 'SB', 'SO', 'HBP', 'SF', 'WAR']
    
    projections = []
    for player in final_active['Player'].unique():
        player_df = final_active[final_active['Player']==player]
        rates_player = rates_active[rates_active['Player']==player]
        
        rates_X = rates_player[['Age', 'G', 'Season_num', 'Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 
                                'Pos_8', 'Pos_9', 'Cluster_0_Prob', 'Cluster_1_Prob',  
                                'BA', 'OPS']]
    
        pred_pa = model.predict(rates_X)
        pred_pa = int(round(pred_pa[0], 0))
    
        player_proj = [player]
        
        for target in targets:
            curr_val = player_df[target].iloc[0]
            rate_val = rates_player[f"{target}_per_500_PA"].iloc[0]
            total_target = curr_val + (rate_val * (pred_pa/500))
            if target != 'WAR':
                total_target = int(total_target)
            else:
                total_target = round(total_target, 1)
            player_proj.append(total_target)
    
        player_proj = player_proj + player_df[['Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 
                        'Pos_8', 'Pos_9', 'GGs', 'SSs', 'All_Stars', 'MVPs', 
                        'Cluster_0_Prob', 'Cluster_1_Prob', 'BA', 'OPS']].iloc[0].tolist()
    
        projections.append(player_proj)
    
    final_cols = ['Player', '2B', '3B', 'HR', 'RBI', 'SB', 'SO', 'HBP', 'SF', 'WAR',
                   'Pos_2', 'Pos_5', 'Pos_6', 'Pos_7', 'Pos_8', 'Pos_9', 'GGs', 'SSs', 
                  'All_Stars', 'MVPs', 'Cluster_0_Prob', 'Cluster_1_Prob', 'BA', 'OPS']
    
    active_projections = pd.DataFrame(data=projections, columns=final_cols)
    
    inactive_cols = final_cols + ['status']
    final_inactive = final_inactive[inactive_cols]
    return active_projections, final_inactive


def classification_preds(final_inactive):
    '''
    
    final_inactive: career total inactive dataset (pd.DataFrame)

    A HOF predictor model is created and trained on the historical data. The model is 
    cross-validated and its performance metrics are printed.

    return: hof predictor model (RandomForestClassifier + BorderlineSMOTE)
    
    '''
    final_inactive_X = final_inactive.iloc[:, 1:24]
    
    y = final_inactive['status'].map({'hof': 1, 'retired': 0})
    
    hof_pipeline = make_pipeline(
        BorderlineSMOTE(sampling_strategy='auto', kind='borderline-1'),
        RandomForestClassifier(class_weight='balanced')
    )

    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(final_inactive_X, y):
        X_train, X_test = final_inactive_X.iloc[train_idx], final_inactive_X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        hof_pipeline.fit(X_train, y_train)
        y_pred = hof_pipeline.predict(X_test)

        precision = precision_score(y_test, y_pred)
        precision_scores.append(precision)
        recall = recall_score(y_test, y_pred)
        recall_scores.append(recall)
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

    print(f"Average Precision: {sum(precision_scores)/len(precision_scores):.3f}")
    print(f"Average Recall: {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"Average F1 Score:  {sum(f1_scores)/len(f1_scores):.3f}")

    return hof_pipeline


def get_active_hofs(active, hof_pipeline):
    '''
    
    active: end of career projections for active players (pd.DataFrame)
    hof predictor model (RandomForestClassifier + BorderlineSMOTE)

    Using the HOF classification model and the end of career projections for active players,
    a dictionary of players, whose average HOF likelihood is validated for, is created. This
    dictionary contains a list of currently active players whose statistical trajectories identify
    as being on a HOF-caliber. 

    return: players of HOF trajectory (dict)
    
    '''
    players = {}
    
    for i in range(5):
        temp = active.copy()
        active_probs = hof_pipeline.predict_proba(temp.iloc[:,1:])
        temp['hof_probability'] = active_probs[:, 1]

        sample = temp[(temp['hof_probability'] >= 0.60) &
                            (temp['WAR'] >= 60.0)]
        for j in sample.index:
            if sample.loc[j, 'Player'] not in players:
                players[sample.loc[j, 'Player']] = [sample.loc[j, 'hof_probability']]
            else:
                players[sample.loc[j, 'Player']].append(sample.loc[j, 'hof_probability'])

    final_players = {}
    for x in players:
        avg_likelihood = round(sum(players[x])/len(players[x]), 2)
        if avg_likelihood >= 0.60:
            final_players[x] = float(avg_likelihood)

    return final_players


main()