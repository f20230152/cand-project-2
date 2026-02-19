from dataclasses import dataclass, field
from typing import Callable, List, Dict

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


@dataclass
class WalkforwardBacktester:
    """
    pnl_df = returns dataframe

    """
    pnl_df: pd.DataFrame
    lookback_period: str  # Lookback period as a frequency string, e.g., '2Y'
    rebalance_freq: str
    score_func: Callable[[pd.Series], float] = field(default=None)
    periods: List = None
    lookback_start_skip_check: bool = False


    # Class variables to store details of each rebalance period
    rebalances: List[Dict[str, any]] = field(default_factory=list)

    def __post_init__(self):
        self.score_func = self.score_func or self._sharpe_ratio
        self.portfolio_pnl = pd.Series(index=self.pnl_df.index, dtype=np.float64)

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        return returns.mean() / returns.std() if returns.std() != 0 else 0

    def _select_best_strategy(self, lookback_df: pd.DataFrame) -> str:
        strategy_scores = lookback_df.apply(self.score_func)
        return strategy_scores.idxmax(), strategy_scores

    def run_backtest(self) -> pd.Series:
        assert self.rebalance_freq[-1] == "M", "rebalance freq need to be in months(M)"
        assert self.lookback_period[-1] == "M", "lookback period need to be in months(M)"
        self.periods = pd.date_range(start=self.pnl_df.index[0],
                                     end=self.pnl_df.index[-1] + relativedelta(months=int(self.rebalance_freq[:-1])),
                                     freq=self.rebalance_freq)
        for start_date, end_date in zip(self.periods[:-1], self.periods[1:]):
            lookback_end = start_date - pd.Timedelta(days=1)  # Lookback ends 1 day before start_date
            lookback_start = lookback_end - relativedelta(months=int(self.lookback_period[:-1]))

            if (lookback_start < self.pnl_df.index[0]) and (not self.lookback_start_skip_check):
                continue  # Skip if lookback start is before the available data

            lookback_df = self.pnl_df.loc[lookback_start:lookback_end]
            best_strat, strategy_scores = self._select_best_strategy(lookback_df)

            # Store details for this rebalance period
            self.rebalances.append({
                'start_date': start_date,
                'end_date': end_date,
                'lookback_start': lookback_start,
                'lookback_end': lookback_end,
                'best_strategy': best_strat,
                'strategy_scores': strategy_scores,  # Store scores for all strategies
                'lookback_df': lookback_df,
                'selected_strategy_mean_returns': self.pnl_df[best_strat].loc[start_date:end_date].mean(),
                'selected_strategy_mean_std': self.pnl_df[best_strat].loc[start_date:end_date].std()
            })

            # Apply selected strategy to the portfolio
            self.portfolio_pnl.loc[start_date:end_date] = self.pnl_df[best_strat].loc[start_date:end_date]

        return self.portfolio_pnl

    def get_rebalance_details(self) -> List[Dict[str, any]]:
        return pd.DataFrame(self.rebalances)
