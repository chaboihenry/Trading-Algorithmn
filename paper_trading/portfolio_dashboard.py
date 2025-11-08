import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioDashboard:
    """
    Dashboard for monitoring paper trading performance

    Database: /Volumes/Vault/85_assets_prediction.db
    Tables: paper_trading_positions, paper_trading_performance

    Features:
    - Portfolio summary and current status
    - Trade history and analytics
    - Performance metrics and charts
    - Position-by-position breakdown
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize dashboard"""
        self.db_path = db_path
        logger.info(f"Initialized PortfolioDashboard")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def get_portfolio_overview(self) -> Dict:
        """Get current portfolio overview with key metrics"""
        conn = self._get_db_connection()

        # Latest performance
        perf_query = """
            SELECT
                total_capital,
                cumulative_pnl,
                open_positions_count,
                closed_positions_count,
                win_rate,
                performance_date
            FROM paper_trading_performance
            ORDER BY performance_date DESC
            LIMIT 1
        """
        perf_df = pd.read_sql(perf_query, conn)

        # Initial capital (first performance record or default)
        initial_query = """
            SELECT total_capital - cumulative_pnl as initial_capital
            FROM paper_trading_performance
            ORDER BY performance_date ASC
            LIMIT 1
        """
        initial_df = pd.read_sql(initial_query, conn)
        initial_capital = initial_df.iloc[0]['initial_capital'] if not initial_df.empty else 1000.0

        # Open positions
        open_query = """
            SELECT
                COUNT(*) as count,
                SUM(capital_allocated) as total_allocated
            FROM paper_trading_positions
            WHERE status = 'OPEN'
        """
        open_df = pd.read_sql(open_query, conn)

        # Closed positions statistics
        closed_query = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losers,
                SUM(realized_pnl) as total_realized_pnl,
                AVG(realized_pnl) as avg_pnl,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade,
                AVG(JULIANDAY(exit_date) - JULIANDAY(entry_date)) as avg_hold_days
            FROM paper_trading_positions
            WHERE status = 'CLOSED'
        """
        closed_df = pd.read_sql(closed_query, conn)

        conn.close()

        # Compile overview
        overview = {}

        if not perf_df.empty:
            p = perf_df.iloc[0]
            overview['total_capital'] = p['total_capital']
            overview['cumulative_pnl'] = p['cumulative_pnl']
            overview['return_pct'] = (p['cumulative_pnl'] / initial_capital) * 100
            overview['open_positions'] = p['open_positions_count']
            overview['closed_positions'] = p['closed_positions_count']
            overview['win_rate'] = p['win_rate'] * 100 if p['win_rate'] else 0
            overview['last_update'] = p['performance_date']
        else:
            overview['total_capital'] = initial_capital
            overview['cumulative_pnl'] = 0
            overview['return_pct'] = 0
            overview['open_positions'] = 0
            overview['closed_positions'] = 0
            overview['win_rate'] = 0
            overview['last_update'] = 'N/A'

        if not open_df.empty:
            overview['capital_allocated'] = open_df.iloc[0]['total_allocated'] or 0
            overview['available_capital'] = overview['total_capital'] - overview['capital_allocated']
        else:
            overview['capital_allocated'] = 0
            overview['available_capital'] = overview['total_capital']

        if not closed_df.empty and closed_df.iloc[0]['total_trades'] > 0:
            c = closed_df.iloc[0]
            overview['total_trades'] = int(c['total_trades'])
            overview['winners'] = int(c['winners'])
            overview['losers'] = int(c['losers'])
            overview['total_realized_pnl'] = c['total_realized_pnl']
            overview['avg_pnl_per_trade'] = c['avg_pnl']
            overview['best_trade'] = c['best_trade']
            overview['worst_trade'] = c['worst_trade']
            overview['avg_hold_days'] = c['avg_hold_days']
        else:
            overview['total_trades'] = 0
            overview['winners'] = 0
            overview['losers'] = 0
            overview['total_realized_pnl'] = 0
            overview['avg_pnl_per_trade'] = 0
            overview['best_trade'] = 0
            overview['worst_trade'] = 0
            overview['avg_hold_days'] = 0

        return overview

    def get_open_positions(self) -> pd.DataFrame:
        """Get all currently open positions with unrealized P&L"""
        conn = self._get_db_connection()

        query = """
            SELECT
                position_id,
                symbol_ticker_1 || '-' || symbol_ticker_2 as pair,
                position_type,
                entry_date,
                entry_z_score,
                entry_price_1,
                entry_price_2,
                shares_1,
                shares_2,
                capital_allocated,
                hedge_ratio,
                CAST(JULIANDAY('now') - JULIANDAY(entry_date) AS INTEGER) as days_held
            FROM paper_trading_positions
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            return df

        # Add current prices and unrealized P&L
        import yfinance as yf

        df['current_price_1'] = 0.0
        df['current_price_2'] = 0.0
        df['unrealized_pnl'] = 0.0
        df['unrealized_pnl_pct'] = 0.0

        for idx, row in df.iterrows():
            # Extract tickers
            tickers = row['pair'].split('-')

            try:
                # Get current prices
                price1 = yf.Ticker(tickers[0]).history(period='1d')['Close'].iloc[-1]
                price2 = yf.Ticker(tickers[1]).history(period='1d')['Close'].iloc[-1]

                df.at[idx, 'current_price_1'] = price1
                df.at[idx, 'current_price_2'] = price2

                # Calculate unrealized P&L
                entry_val1 = row['shares_1'] * row['entry_price_1']
                entry_val2 = row['shares_2'] * row['entry_price_2']
                current_val1 = row['shares_1'] * price1
                current_val2 = row['shares_2'] * price2

                if row['position_type'] == 'Long_Spread':
                    pnl = (current_val1 - entry_val1) - (current_val2 - entry_val2)
                else:  # Short_Spread
                    pnl = (entry_val1 - current_val1) + (current_val2 - entry_val2)

                df.at[idx, 'unrealized_pnl'] = pnl
                df.at[idx, 'unrealized_pnl_pct'] = (pnl / row['capital_allocated']) * 100

            except Exception as e:
                logger.warning(f"Could not fetch prices for {row['pair']}: {str(e)}")

        return df

    def get_closed_positions(self, limit: int = 20) -> pd.DataFrame:
        """Get recent closed positions"""
        conn = self._get_db_connection()

        query = f"""
            SELECT
                position_id,
                symbol_ticker_1 || '-' || symbol_ticker_2 as pair,
                position_type,
                entry_date,
                exit_date,
                entry_z_score,
                exit_z_score,
                realized_pnl,
                ROUND((realized_pnl / capital_allocated) * 100, 2) as return_pct,
                exit_reason,
                CAST(JULIANDAY(exit_date) - JULIANDAY(entry_date) AS INTEGER) as days_held
            FROM paper_trading_positions
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC
            LIMIT {limit}
        """

        df = pd.read_sql(query, conn)
        conn.close()

        return df

    def get_performance_history(self) -> pd.DataFrame:
        """Get performance history over time"""
        conn = self._get_db_connection()

        query = """
            SELECT
                performance_date,
                total_capital,
                cumulative_pnl,
                daily_pnl,
                open_positions_count,
                closed_positions_count,
                ROUND(win_rate * 100, 2) as win_rate_pct
            FROM paper_trading_performance
            ORDER BY performance_date ASC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        return df

    def get_trade_statistics_by_pair(self) -> pd.DataFrame:
        """Get statistics grouped by pair"""
        conn = self._get_db_connection()

        query = """
            SELECT
                symbol_ticker_1 || '-' || symbol_ticker_2 as pair,
                COUNT(*) as trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                ROUND(AVG(realized_pnl), 2) as avg_pnl,
                ROUND(SUM(realized_pnl), 2) as total_pnl,
                MAX(realized_pnl) as best_trade,
                MIN(realized_pnl) as worst_trade
            FROM paper_trading_positions
            WHERE status = 'CLOSED'
            GROUP BY symbol_ticker_1, symbol_ticker_2
            ORDER BY total_pnl DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            df['win_rate'] = (df['winners'] / df['trades'] * 100).round(2)

        return df

    def get_trade_statistics_by_signal(self) -> pd.DataFrame:
        """Get statistics by position type (Long_Spread vs Short_Spread)"""
        conn = self._get_db_connection()

        query = """
            SELECT
                position_type,
                COUNT(*) as trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winners,
                ROUND(AVG(realized_pnl), 2) as avg_pnl,
                ROUND(SUM(realized_pnl), 2) as total_pnl,
                ROUND(AVG(JULIANDAY(exit_date) - JULIANDAY(entry_date)), 1) as avg_days_held
            FROM paper_trading_positions
            WHERE status = 'CLOSED'
            GROUP BY position_type
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            df['win_rate'] = (df['winners'] / df['trades'] * 100).round(2)

        return df

    def print_dashboard(self) -> None:
        """Print comprehensive dashboard to console"""
        print(f"\n{'='*80}")
        print(f"{'PAIRS TRADING PAPER PORTFOLIO DASHBOARD':^80}")
        print(f"{'='*80}\n")

        # Portfolio Overview
        overview = self.get_portfolio_overview()

        print(f"{'PORTFOLIO SUMMARY':^80}")
        print(f"{'-'*80}")
        print(f"Total Capital:        ${overview['total_capital']:>12,.2f}")
        print(f"Initial Capital:      ${overview['total_capital'] - overview['cumulative_pnl']:>12,.2f}")
        print(f"Cumulative P&L:       ${overview['cumulative_pnl']:>12,.2f}  ({overview['return_pct']:>6.2f}%)")
        print(f"Available Capital:    ${overview['available_capital']:>12,.2f}")
        print(f"Allocated Capital:    ${overview['capital_allocated']:>12,.2f}")
        print(f"Last Updated:         {overview['last_update']:>20}")
        print(f"{'-'*80}\n")

        # Trading Statistics
        print(f"{'TRADING STATISTICS':^80}")
        print(f"{'-'*80}")
        print(f"Total Trades:         {overview['total_trades']:>12}")
        print(f"Open Positions:       {overview['open_positions']:>12}")
        print(f"Closed Positions:     {overview['closed_positions']:>12}")
        print(f"Winners:              {overview['winners']:>12}")
        print(f"Losers:               {overview['losers']:>12}")
        print(f"Win Rate:             {overview['win_rate']:>11.2f}%")
        print(f"{'-'*80}")
        print(f"Total Realized P&L:   ${overview['total_realized_pnl']:>12,.2f}")
        print(f"Avg P&L per Trade:    ${overview['avg_pnl_per_trade']:>12,.2f}")
        print(f"Best Trade:           ${overview['best_trade']:>12,.2f}")
        print(f"Worst Trade:          ${overview['worst_trade']:>12,.2f}")
        print(f"Avg Hold Time:        {overview['avg_hold_days']:>11.1f} days")
        print(f"{'-'*80}\n")

        # Open Positions
        open_positions = self.get_open_positions()
        if not open_positions.empty:
            print(f"{'OPEN POSITIONS':^80}")
            print(f"{'-'*80}")
            for idx, pos in open_positions.iterrows():
                print(f"\n{pos['pair']} ({pos['position_type']})")
                print(f"  Entry Date:     {pos['entry_date']}")
                print(f"  Entry Z-score:  {pos['entry_z_score']:.2f}")
                print(f"  Days Held:      {pos['days_held']}")
                print(f"  Capital:        ${pos['capital_allocated']:,.2f}")
                print(f"  Unrealized P&L: ${pos['unrealized_pnl']:,.2f} ({pos['unrealized_pnl_pct']:.2f}%)")
            print(f"\n{'-'*80}\n")
        else:
            print(f"{'NO OPEN POSITIONS':^80}\n")

        # Recent Closed Positions
        closed_positions = self.get_closed_positions(limit=10)
        if not closed_positions.empty:
            print(f"{'RECENT CLOSED POSITIONS (Last 10)':^80}")
            print(f"{'-'*80}")
            print(closed_positions[['pair', 'entry_date', 'exit_date', 'realized_pnl',
                                   'return_pct', 'exit_reason', 'days_held']].to_string(index=False))
            print(f"{'-'*80}\n")

        # Performance by Pair
        pair_stats = self.get_trade_statistics_by_pair()
        if not pair_stats.empty:
            print(f"{'PERFORMANCE BY PAIR':^80}")
            print(f"{'-'*80}")
            print(pair_stats[['pair', 'trades', 'total_pnl', 'avg_pnl',
                            'win_rate']].head(10).to_string(index=False))
            print(f"{'-'*80}\n")

        # Performance by Signal Type
        signal_stats = self.get_trade_statistics_by_signal()
        if not signal_stats.empty:
            print(f"{'PERFORMANCE BY SIGNAL TYPE':^80}")
            print(f"{'-'*80}")
            print(signal_stats.to_string(index=False))
            print(f"{'-'*80}\n")

        # Performance History
        perf_history = self.get_performance_history()
        if not perf_history.empty and len(perf_history) > 1:
            print(f"{'PERFORMANCE HISTORY':^80}")
            print(f"{'-'*80}")
            print(perf_history.tail(10).to_string(index=False))
            print(f"{'-'*80}\n")

        print(f"{'='*80}\n")

    def export_to_csv(self, output_dir: str = ".") -> None:
        """Export all data to CSV files for analysis"""
        import os

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export open positions
        open_pos = self.get_open_positions()
        if not open_pos.empty:
            filename = os.path.join(output_dir, f"open_positions_{timestamp}.csv")
            open_pos.to_csv(filename, index=False)
            logger.info(f"Exported open positions to {filename}")

        # Export closed positions
        closed_pos = self.get_closed_positions(limit=1000)
        if not closed_pos.empty:
            filename = os.path.join(output_dir, f"closed_positions_{timestamp}.csv")
            closed_pos.to_csv(filename, index=False)
            logger.info(f"Exported closed positions to {filename}")

        # Export performance history
        perf_history = self.get_performance_history()
        if not perf_history.empty:
            filename = os.path.join(output_dir, f"performance_history_{timestamp}.csv")
            perf_history.to_csv(filename, index=False)
            logger.info(f"Exported performance history to {filename}")

        print(f"\nData exported successfully to {output_dir}/")


if __name__ == "__main__":
    # Initialize dashboard
    dashboard = PortfolioDashboard()

    # Print comprehensive dashboard
    dashboard.print_dashboard()

    # Optional: Export to CSV
    # dashboard.export_to_csv()
