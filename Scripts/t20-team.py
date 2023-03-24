import numpy as np


class Team:
    def __init__(self, name, bat_order, bat_lookup, bwl_lookup, mtc_ply_df):
        self.name = name  # team name
        self.bat_lookup = bat_lookup  # batter stats dictionary
        self.bwl_lookup = bwl_lookup  # bowler stats dictionary
        self.mtc_ply_df = mtc_ply_df  # historical player statistic (pandas dataframe)
        self.order = {i + 1: x for i, x in enumerate(bat_order)}  # batting order
        self.p_header = ["W", "0", "1", "2", "3", "4", "6", "WD"]

        # batting innings state
        self.bat_extras = 0
        self.bat_total = 0
        self.bat_wkts = 0

        # bowling innings state
        self.bwl_extras = 0
        self.bwl_total = 0
        self.bwl_wkts = 0

        # player match statistics
        self.ply_stats = dict()
        for i, ply in enumerate(bat_order):
            self.ply_stats[ply] = {
                "runs": 0,
                "balls": 0,
                "4s": 0,
                "6s": 0,
                "out": 0,
                "overs": 0,
                "maidens": 0,
                "runs_off": 0,
                "wickets": 0,
                "wides": 0,
                "bat_order": i + 1,
            }
            # calculate historical average # overs bowled for each player
            if ply in mtc_ply_df.player_id.unique():
                self.ply_stats[ply]["ave_overs"] = self.mtc_ply_df[
                    self.mtc_ply_df.player_id == ply
                ].overs.mean()
            elif (i + 1 >= 7) and (i + 1 <= 9):
                self.ply_stats[ply]["ave_overs"] = 2.5
            elif i + 1 >= 10:
                self.ply_stats[ply]["ave_overs"] = 4
            else:
                self.ply_stats[ply]["ave_overs"] = 0

        # initialise team state
        self.onstrike = self.order[1]
        self.offstrike = self.order[2]
        self.bowler = self.nxt_bowler(first_over=True)
        self.bat_bwl = ""

    def get_probs(self, pid, stats):
        # module to return historical stats for a given player
        if stats == "bat":
            if pid in self.bat_lookup:
                ps = self.bat_lookup[pid]
            else:
                ps = self.bat_lookup["unknown"]
        elif stats == "bwl":
            if pid in self.bwl_lookup:
                ps = self.bwl_lookup[pid]
            else:
                ps = self.bwl_lookup["unknown"]
        else:
            return None
        return [ps[x] for x in self.p_header]

    def nxt_bowler(self, first_over=False):
        # module to choose next bowler
        if first_over:
            lst_bwler = ""
        else:
            lst_bwler = self.bowler

        pids = []
        probs = []
        for pid in self.ply_stats:
            if (pid != lst_bwler) and (self.ply_stats[pid]["overs"] < 4):
                pids.append(pid)
                probs.append(self.ply_stats[pid]["ave_overs"])

        return np.random.choice(a=pids, size=1, p=np.array(probs) / sum(probs))[0]

    def wicket(self):
        # module for updating the team after a wicket
        self.bat_wkts += 1
        self.ply_stats[self.onstrike]["out"] = 1
        if self.bat_wkts < 10:
            self.onstrike = self.order[self.bat_wkts + 2]

    def new_over(self):
        # module to start a new over
        if self.bat_bwl == "bat":
            self.change_ends()
        elif self.bat_bwl == "bwl":
            self.bowler = self.nxt_bowler()
            self.ply_stats[self.bowler]["overs"] += 1

    def change_ends(self):
        # module to change the on-strike batter between overs
        temp = self.onstrike
        self.onstrike = self.offstrike
        self.offstrike = temp
