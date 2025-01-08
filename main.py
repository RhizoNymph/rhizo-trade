import argparse
from index.velo.client import handle_velo_spot, handle_velo_futures, handle_velo_options
from index.geckoterminal.client import handle_geckoterminal
from index.fred.client import handle_fred

def add_common_velo_args(parser):
    """Add arguments that are common to all Velo commands"""
    parser.add_argument(
        '--resolution',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d'],
        help='Data resolution'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        help='Exchange to index (use "all" for all exchanges)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=24*180,
        help='Hours of historical data to fetch (default: 24)'
    )

def main():
    parser = argparse.ArgumentParser(description='Market Data Tool')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index market data')
    index_subparsers = index_parser.add_subparsers(dest='index_type', help='Type of indexing')

    # GeckoTerminal
    gecko_parser = index_subparsers.add_parser('geckoterminal', help='Index GeckoTerminal data')
    gecko_parser.add_argument(
        '--resolution',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '1h', '4h', '12h', '1d'],
        help='Data resolution'
    )
    gecko_parser.add_argument(
        '--network',
        type=str,
        required=True,
        choices=['eth', 'bsc', 'polygon', 'arbitrum', 'optimism', 'avalanche', 'fantom', 'celo', 'solana', 'base', 'osmosis'],
        help='Network to index'
    )
    gecko_parser.add_argument(
        '--pool-type',
        type=str,
        default='trending',
        choices=['trending', 'top', 'new'],
        help='Type of pools to fetch (default: trending)'
    )
    gecko_parser.add_argument(
        '--lookback',
        type=int,
        default=24,
        help='Hours of historical data to fetch (default: 24)'
    )

    # Velo
    velo_parser = index_subparsers.add_parser('velo', help='Index Velo data')
    velo_subparsers = velo_parser.add_subparsers(dest='velo_type', help='Type of Velo data')

    # Velo Spot
    velo_spot_parser = velo_subparsers.add_parser('spot', help='Index Velo spot data')
    add_common_velo_args(velo_spot_parser)

    # Velo Futures
    velo_futures_parser = velo_subparsers.add_parser('futures', help='Index Velo futures data')
    add_common_velo_args(velo_futures_parser)

    # Velo Options
    velo_options_parser = velo_subparsers.add_parser('options', help='Index Velo options data')
    add_common_velo_args(velo_options_parser)

    # FRED
    fred_parser = index_subparsers.add_parser('fred', help='Index FRED data')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze market data')
    # Add analyze-specific arguments here if needed

    visualize_parser = subparsers.add_parser('visualize', help='Visualize market data')
    visualize_subparsers = visualize_parser.add_subparsers(dest='viz_type', help='Type of visualization')

    # Regression visualization
    regression_parser = visualize_subparsers.add_parser('regression', help='Run regression analysis dashboard')

    args = parser.parse_args()

    # Handle commands
    try:
        if args.mode == 'index':
            if args.index_type == 'geckoterminal':
                handle_geckoterminal(args)
            elif args.index_type == 'velo':
                if args.velo_type == 'spot':
                    handle_velo_spot(args)
                elif args.velo_type == 'futures':
                    handle_velo_futures(args)
                elif args.velo_type == 'options':
                    handle_velo_options(args)
                else:
                    print("Please specify a Velo data type (spot/futures/options)")
            elif args.index_type == 'fred':
                handle_fred(args)
            else:
                print("Please specify an index type (geckoterminal/velo)")
        elif args.mode == 'analyze':
            print("Analyze functionality not implemented yet")
        elif args.mode == 'visualize':
            if args.viz_type == 'regression':
                run_regression_dashboard(                )
        else:
            print("Please specify a mode (index/analyze/visualize)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
