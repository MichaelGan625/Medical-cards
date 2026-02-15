"""
MSS Configuration Comparison
比较不同配置下的MSS表现，生成对比报告
"""

import json
import argparse
from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(filepath: str) -> Dict:
    """加载网格搜索结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_configs(results: List[Dict], config_names: List[str]) -> pd.DataFrame:
    """比较多个配置"""
    comparison_data = []
    
    for config, name in zip(results, config_names):
        comparison_data.append({
            'config_name': name,
            'lambda_mmr': config.get('lambda_mmr', 'N/A'),
            'w_redundancy': config.get('w_redundancy', 'N/A'),
            'w_base': config.get('w_base', 'N/A'),
            'coverage': config['coverage'],
            'redundancy': config['redundancy'],
            'diversity': config['diversity'],
            'avg_base_score': config['avg_base_score'],
            'overall': config['overall'],
            'time_hours': config.get('time_hours', 'N/A')
        })
    
    return pd.DataFrame(comparison_data)


def create_comparison_viz(df: pd.DataFrame, output_file: str):
    """创建对比可视化"""
    
    # 创建雷达图比较
    metrics = ['coverage', 'diversity', 'avg_base_score', 'overall']
    metric_labels = ['Coverage', 'Diversity', 'Avg Base Score', 'Overall']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Coverage Comparison', 'Diversity Comparison', 
                       'Quality vs Diversity', 'Overall Score Comparison'],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Coverage对比
    fig.add_trace(
        go.Bar(x=df['config_name'], y=df['coverage'], 
               name='Coverage', marker_color=colors[0],
               text=df['coverage'].round(4), textposition='outside'),
        row=1, col=1
    )
    
    # Diversity对比
    fig.add_trace(
        go.Bar(x=df['config_name'], y=df['diversity'], 
               name='Diversity', marker_color=colors[1],
               text=df['diversity'].round(4), textposition='outside'),
        row=1, col=2
    )
    
    # Quality vs Diversity散点图
    fig.add_trace(
        go.Scatter(
            x=df['diversity'], y=df['avg_base_score'],
            mode='markers+text',
            text=df['config_name'],
            textposition='top center',
            marker=dict(size=15, color=colors[2]),
            name='Quality-Diversity Trade-off'
        ),
        row=2, col=1
    )
    
    # Overall对比
    fig.add_trace(
        go.Bar(x=df['config_name'], y=df['overall'], 
               name='Overall', marker_color=colors[3],
               text=df['overall'].round(4), textposition='outside'),
        row=2, col=2
    )
    
    # 更新布局
    fig.update_xaxes(title_text="Configuration", row=1, col=1)
    fig.update_xaxes(title_text="Configuration", row=1, col=2)
    fig.update_xaxes(title_text="Diversity", row=2, col=1)
    fig.update_xaxes(title_text="Configuration", row=2, col=2)
    
    fig.update_yaxes(title_text="Coverage", row=1, col=1, range=[0.85, 1.0])
    fig.update_yaxes(title_text="Diversity", row=1, col=2, range=[0, 0.3])
    fig.update_yaxes(title_text="Avg Base Score", row=2, col=1, range=[0.7, 1.0])
    fig.update_yaxes(title_text="Overall Score", row=2, col=2)
    
    fig.update_layout(
        title_text="MSS Configuration Comparison",
        height=900,
        showlegend=False
    )
    
    fig.write_html(output_file)
    print(f"✅ Comparison visualization saved: {output_file}")


def generate_report(df: pd.DataFrame, output_file: str):
    """生成文本报告"""
    
    report_lines = [
        "="*80,
        "MSS Configuration Comparison Report",
        "="*80,
        "",
    ]
    
    # 配置参数对比
    report_lines.extend([
        "📋 Configuration Parameters:",
        "-"*80,
    ])
    
    for _, row in df.iterrows():
        report_lines.extend([
            f"\n🎯 {row['config_name']}:",
            f"   Lambda MMR: {row['lambda_mmr']}",
            f"   W Redundancy: {row['w_redundancy']}",
            f"   W Base Score: {row['w_base']}",
        ])
    
    # 性能指标对比
    report_lines.extend([
        "",
        "",
        "📊 Performance Metrics:",
        "-"*80,
        "",
    ])
    
    # 创建对比表格
    table_data = df[['config_name', 'coverage', 'redundancy', 'diversity', 
                     'avg_base_score', 'overall']].copy()
    table_data.columns = ['Config', 'Coverage', 'Redundancy', 'Diversity', 
                          'Avg Base', 'Overall']
    
    report_lines.append(table_data.to_string(index=False))
    
    # 最佳配置
    report_lines.extend([
        "",
        "",
        "🏆 Best Configurations:",
        "-"*80,
    ])
    
    best_overall = df.loc[df['overall'].idxmax()]
    best_coverage = df.loc[df['coverage'].idxmax()]
    best_diversity = df.loc[df['diversity'].idxmax()]
    best_quality = df.loc[df['avg_base_score'].idxmax()]
    
    report_lines.extend([
        f"\n• Best Overall Score: {best_overall['config_name']} "
        f"(Overall={best_overall['overall']:.4f})",
        f"  λ={best_overall['lambda_mmr']}, w_red={best_overall['w_redundancy']}, "
        f"w_base={best_overall['w_base']}",
        "",
        f"• Best Coverage: {best_coverage['config_name']} "
        f"(Coverage={best_coverage['coverage']:.4f})",
        "",
        f"• Best Diversity: {best_diversity['config_name']} "
        f"(Diversity={best_diversity['diversity']:.4f})",
        "",
        f"• Best Quality: {best_quality['config_name']} "
        f"(Avg Base={best_quality['avg_base_score']:.4f})",
    ])
    
    # 改进分析
    if 'Baseline' in df['config_name'].values and len(df) > 1:
        baseline = df[df['config_name'] == 'Baseline'].iloc[0]
        report_lines.extend([
            "",
            "",
            "📈 Improvement Analysis (vs Baseline):",
            "-"*80,
        ])
        
        for _, row in df.iterrows():
            if row['config_name'] == 'Baseline':
                continue
            
            cov_change = (row['coverage'] - baseline['coverage']) / baseline['coverage'] * 100
            div_change = (row['diversity'] - baseline['diversity']) / baseline['diversity'] * 100
            red_change = (row['redundancy'] - baseline['redundancy']) / baseline['redundancy'] * 100
            overall_change = (row['overall'] - baseline['overall']) / baseline['overall'] * 100
            
            report_lines.extend([
                f"\n{row['config_name']}:",
                f"  Coverage: {cov_change:+.2f}%",
                f"  Diversity: {div_change:+.2f}% ⬆" if div_change > 0 else f"  Diversity: {div_change:+.2f}% ⬇",
                f"  Redundancy: {red_change:+.2f}% ⬇" if red_change < 0 else f"  Redundancy: {red_change:+.2f}% ⬆",
                f"  Overall: {overall_change:+.2f}%",
            ])
    
    # 推荐配置
    report_lines.extend([
        "",
        "",
        "💡 Recommendations:",
        "-"*80,
    ])
    
    # 根据多样性改进推荐
    diversity_improved = df[df['diversity'] > df['diversity'].quantile(0.5)]
    if not diversity_improved.empty:
        top_diverse = diversity_improved.nlargest(1, 'overall').iloc[0]
        report_lines.extend([
            f"\n✅ Recommended Configuration: {top_diverse['config_name']}",
            f"   λ={top_diverse['lambda_mmr']}, w_red={top_diverse['w_redundancy']}, "
            f"w_base={top_diverse['w_base']}",
            f"   • Coverage: {top_diverse['coverage']:.4f}",
            f"   • Diversity: {top_diverse['diversity']:.4f}",
            f"   • Redundancy: {top_diverse['redundancy']:.4f}",
            f"   • Quality: {top_diverse['avg_base_score']:.4f}",
            f"   • Overall: {top_diverse['overall']:.4f}",
            "",
            "   Rationale: Balances high diversity with strong overall performance.",
        ])
    
    report_lines.extend([
        "",
        "="*80,
        "Report generated successfully!",
        "="*80,
    ])
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Text report saved: {output_file}")
    
    # 同时打印到控制台
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Compare MSS configurations")
    parser.add_argument("--baseline", type=str, required=True, 
                       help="Path to baseline results JSON")
    parser.add_argument("--experiments", type=str, nargs="+", required=True,
                       help="Paths to experimental results JSONs")
    parser.add_argument("--config_names", type=str, nargs="+", 
                       default=None,
                       help="Names for each configuration")
    parser.add_argument("--k", type=int, default=800,
                       help="K value to compare (default: 800)")
    parser.add_argument("--output_html", default="mss_comparison_viz.html")
    parser.add_argument("--output_report", default="mss_comparison_report.txt")
    
    args = parser.parse_args()
    
    # 加载基线结果
    print(f"📂 Loading baseline: {args.baseline}")
    baseline_data = load_results(args.baseline)
    
    # 加载实验结果
    experiment_data = []
    for exp_file in args.experiments:
        print(f"📂 Loading experiment: {exp_file}")
        experiment_data.append(load_results(exp_file))
    
    # 提取K=800的结果
    all_results = [baseline_data] + experiment_data
    k_results = []
    
    for data in all_results:
        if 'results' in data:
            # Grid search format
            matching = [r for r in data['results'] if r.get('k') == args.k]
            if matching:
                # 取Overall最高的
                k_results.append(max(matching, key=lambda x: x['overall']))
        elif 'best_by_k' in data:
            # Method comparison format
            matching = [r for r in data['best_by_k'] if r.get('k') == args.k]
            if matching:
                k_results.append(matching[0])
    
    if len(k_results) != len(all_results):
        print(f"⚠️ Warning: Could not find K={args.k} results in all files")
    
    # 配置名称
    if args.config_names:
        config_names = args.config_names
    else:
        config_names = ['Baseline'] + [f'Experiment {i+1}' for i in range(len(args.experiments))]
    
    if len(config_names) < len(k_results):
        config_names += [f'Config {i+1}' for i in range(len(config_names), len(k_results))]
    
    # 创建对比DataFrame
    df = compare_configs(k_results, config_names[:len(k_results)])
    
    # 生成可视化
    print("\n📊 Generating visualization...")
    create_comparison_viz(df, args.output_html)
    
    # 生成报告
    print("\n📝 Generating text report...")
    generate_report(df, args.output_report)
    
    print(f"\n✨ Comparison complete! Check {args.output_html} and {args.output_report}")


if __name__ == "__main__":
    main()
