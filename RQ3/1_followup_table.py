# Data for the table
data = [
    # Language, Conversations, Single-turn, Multi-turn, 1st-turn code snippets
    ("Python", 22539, 11537, 11001, 32586),
    ("JavaScript", 7385, 4353, 3032, 10128),
    ("C++", 5404, 2780, 2624, 7788),
    ("Java", 4557, 2496, 2061, 6922),
    ("C#", 4355, 2328, 2027, 6261),
]

# LaTeX table header
latex = r"""\begin{table}[ht]
\centering
\caption{Summary of conversation and code snippet statistics for each language.}
\label{tab:conv_stats}
\begin{tabular}{lrrrrr}
\hline
Language & Conversations & Single-turn & Multi-turn & 1st-turn code snippets \\
\hline
"""

# Add rows
for row in data:
    latex += f"{row[0]} & {row[1]:,} & {row[2]:,} & {row[3]:,} & {row[4]:,} \\\\\n"

# Table footer
latex += r"""\hline
\end{tabular}
\end{table}
"""

print(latex)