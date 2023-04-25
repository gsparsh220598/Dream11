import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Functions to calculate derived features, batting
def centuries(striker_data):
    return (striker_data["runs"] >= 100).astype(int)


def fifties(striker_data):
    return ((striker_data["runs"] >= 50) & (striker_data["runs"] < 100)).astype(int)


def zeros(striker_data):
    return (striker_data["runs"] == 0).astype(int)


def highest_score(striker_data):
    return striker_data.groupby("striker")["runs"].max()


def batting_average(striker_data):
    return striker_data.runs.div(striker_data.dismissals, axis=0)


def batting_strike_rate(striker_data):
    return (striker_data.runs / striker_data.balls) * 100


# Functions to calculate derived features, bowling
def num_innings_bowl(df, bowler_col="bowler"):
    return df.groupby(bowler_col).size()


def overs(bowler_df):
    ovs = bowler_df.balls // 6
    if ovs > 4:
        return 4
    else:
        return ovs


def four_wicket_hauls(bowler_df):
    return (bowler_df.wickets >= 4).astype(int)


def bowling_average(bowler_df):
    return bowler_df.runs_conceded.div(bowler_df.wickets, axis=0)


def bowling_strike_rate(bowler_df):
    return bowler_df.balls.div(bowler_df.wickets, axis=0)


def bowling_economy_rate(bowler_df):
    overs = overs(bowler_df)
    runs_conceded = bowler_df.runs_conceded
    economy_rate = runs_conceded / overs
    return economy_rate


# Additional functions to calculate derived features
def batting_consistency(striker_data):
    return (
        0.4262 * striker_data.average
        + 0.2566 * striker_data.shape[0]
        + 0.1510 * striker_data.sr
        + 0.0787 * striker_data.centuries
        + 0.0556 * striker_data.fifties
        - 0.0328 * striker_data.zeros
    )


def bowling_consistency(bowler_data, average, sr, overs, ff):
    return (
        0.4174 * overs
        + 0.2634 * bowler_data.shape[0]
        + 0.1602 * sr
        + 0.0975 * average
        + 0.0615 * ff
    )


def batting_form(striker_data):  # TODO: add date filter (only last 12 months)
    return (
        0.4262 * striker_data.average
        + 0.2566 * striker_data.shape[0]
        + 0.1510 * striker_data.sr
        + 0.0787 * striker_data.centuries
        + 0.0556 * striker_data.fifties
        - 0.0328 * striker_data.zeros
    )


def bowling_form(bowler_data, average, sr, overs, ff):
    return (
        0.3269 * overs
        + 0.2846 * bowler_data.shape[0]
        + 0.1877 * sr
        + 0.1210 * average
        + 0.0798 * ff
    )


def batting_opposition(striker_data):
    return (
        0.4262 * striker_data.average
        + 0.2566 * striker_data.shape[0]
        + 0.1510 * striker_data.sr
        + 0.0787 * striker_data.centuries
        + 0.0556 * striker_data.fifties
        - 0.0328 * striker_data.zeros
    )


def bowling_opposition(bowler_data, average, sr, overs, ff):
    return (
        0.3177 * overs
        + 0.3177 * bowler_data.shape[0]
        + 0.1933 * sr
        + 0.1465 * average
        + 0.0943 * ff
    )


def batting_venue(striker_data):
    return (
        0.4262 * striker_data.average
        + 0.2566 * striker_data.shape[0]
        + 0.1510 * striker_data.sr
        + 0.0787 * striker_data.centuries
        + 0.0556 * striker_data.fifties
        + 0.0328 * striker_data.hs
    )


def bowling_venue(bowler_data, average, sr, overs, ff):
    return (
        0.3018 * overs
        + 0.2783 * bowler_data.shape[0]
        + 0.1836 * sr
        + 0.1391 * average
        + 0.0972 * ff
    )


def prepare_batsman_data(df):
    df_batsman = (
        df.groupby(["striker", "match_id"])
        .agg(
            runs=pd.NamedAgg(column="runs_off_bat", aggfunc="sum"),
            balls=pd.NamedAgg(column="ball", aggfunc="count"),
            innings=pd.NamedAgg(column="match_id", aggfunc="nunique"),
            dismissals=pd.NamedAgg(
                column="wicket_type", aggfunc=lambda x: x.notna().sum()
            ),
        )
        .reset_index()
    )

    df_batsman["centuries"] = centuries(df_batsman)
    df_batsman["fifties"] = fifties(df_batsman)
    df_batsman["zeros"] = zeros(df_batsman)
    df_batsman["average"] = batting_average(df_batsman)
    df_batsman["sr"] = batting_strike_rate(df_batsman)
    df_batsman["hs"] = highest_score(df_batsman)
    df_batsman["consistency"] = batting_consistency(df_batsman)
    df_batsman["form"] = batting_form(df_batsman)
    df_batsman["opposition"] = batting_opposition(df_batsman)
    df_batsman["venue"] = batting_venue(df_batsman)

    return df_batsman


def prepare_bowler_data(df):
    df_bowler = (
        df.groupby(["bowler", "match_id"])
        .agg(
            wickets=pd.NamedAgg(
                column="wicket_type", aggfunc=lambda x: x.notna().sum()
            ),
            balls=pd.NamedAgg(column="ball", aggfunc="count"),
            innings=pd.NamedAgg(column="match_id", aggfunc="nunique"),
            runs_conceded=pd.NamedAgg(column="runs_off_bat", aggfunc="sum"),
        )
        .reset_index()
    )

    df_bowler["overs"] = overs(df_bowler)
    df_bowler["ff"] = four_wicket_hauls(df_bowler)
    df_bowler["average"] = bowling_average(df_bowler)
    df_bowler["sr"] = bowling_strike_rate(df_bowler)
    df_bowler["economy"] = bowling_economy_rate(df_bowler)

    return df_bowler


def create_features(df):
    features = pd.DataFrame()

    # Calculate base statistics
    batsman_stats = prepare_batsman_data(df)
    print(batsman_stats.head())

    bowler_stats = prepare_bowler_data(df)
    print(bowler_stats.head())

    # Calculate recent form statistics
    recent_batsman_stats = (df, "striker")
    recent_bowler_stats = recent_form(df, "bowler")

    # Calculate opposition statistics
    opposition_batsman_stats = opposition_performance(df, "striker")
    opposition_bowler_stats = opposition_performance(df, "bowler")

    # Calculate venue statistics
    venue_batsman_stats = venue_performance(df, "striker")
    venue_bowler_stats = venue_performance(df, "bowler")

    # Combine all the statistics into one DataFrame
    features = pd.concat(
        [
            batsman_stats,
            bowler_stats,
            recent_batsman_stats,
            recent_bowler_stats,
            opposition_batsman_stats,
            opposition_bowler_stats,
            venue_batsman_stats,
            venue_bowler_stats,
        ],
        axis=1,
    )

    # Rename the columns for easier understanding
    features.columns = [
        "batsman_average",
        "batsman_sr",
        "batsman_centuries",
        "batsman_fifties",
        "batsman_zeros",
        "batsman_hs",
        "batsman_innings",
        "bowler_average",
        "bowler_sr",
        "bowler_overs",
        "bowler_ff",
        "bowler_innings",
        "striker_form",
        "bowler_form",
        "batting_opposition",
        "bowling_opposition",
        "batting_venue",
        "bowling_venue",
    ]

    return features


if __name__ == "__main__":
    file_path = "../Inputs/ipl_csv/all_matches.csv"
    # file_path = "../Inputs/temp.csv"
    df = load_data(file_path)
    features = create_features(df[:1000])
    print(features)
