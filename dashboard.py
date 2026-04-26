"""
Decision Tree Solver v4 — Enhanced Dashboard
=============================================
Run:  streamlit run decision_tree_dashboard_v4.py
Deps: pip install streamlit plotly pandas numpy graphviz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import graphviz, math

# ─────────────────────────────────────────────────────────────────
# 1.  TREE DATA
# ─────────────────────────────────────────────────────────────────
TREES = {
    "probabilistic": {
        "id":"D0","type":"decision","label":"D₀","text":"",
        "children":[
            {"id":"C2_D0","type":"chance","label":"C₂","weights":[0.7,0.3],"text":"d01: Changer de strategie","titre":"Concurrence",
             "children":[
                 # ── child 0 of C2_D0 = d₀₁ branch → D₁
                 {"id":"D1","type":"decision","label":"D₁","text":"",
                  "children":[
                      # ── child 0 of D1 = d₁₁ → C1_D1
                      {"id":"C1_D1","type":"chance","label":"C₁","weights":[0.75,0.25],"text":"d11: Renforcer le positionnement en Chine","titre":"Tension geopolitique",
                       "children":[
                           # ── child 0 of C1_D1 = D3
                           {"id":"D3","type":"decision","label":"D₃","text":"",
                            "children":[
                                # child 0 of D3 = d₃₁ → C3_D3
                                {"id":"C3_D3","type":"chance","label":"C₃","weights":[0.4,0.6],"text":"d31: Faire un partenariat avec une marque local","titre":"Demande",
                                 "children":[
                                     {"id":"L1","type":"leaf","label":"+500 M","utility":500,"text":""},
                                     {"id":"L2","type":"leaf","label":"-1.4 Md","utility":-1400,"text":""}]},
                                # child 1 of D3 = d₃₂ → C2_D3
                                {"id":"C2_D3","type":"chance","label":"C₂","weights":[0.6,0.4],"text":"d32: Rebranding local","titre":"Concurrence",
                                 "children":[
                                     {"id":"L3","type":"leaf","label":"-800 M","utility":-800,"text":""},
                                     {"id":"L4","type":"leaf","label":"+700 M","utility":700,"text":""}]}
                            ]},
                           # ── child 1 of C1_D1 = D4
                           {"id":"D4","type":"decision","label":"D₄", "text":"",
                            "children":[
                                # child 0 of D4 = d₄₁ → C3_D4
                                {"id":"C3_D4","type":"chance","label":"C₃","weights":[0.6,0.3,0.1],"text":"d41: Diversifier la capacite de distribution","titre":"Demande",
                                 "children":[
                                     {"id":"L5","type":"leaf","label":"+500 M","utility":500,"text":""},
                                     {"id":"L6","type":"leaf","label":"+200 M","utility":200,"text":""},
                                     {"id":"L7","type":"leaf","label":"-10 M","utility":-10,"text":""}]},
                                # child 1 of D4 = d₄₂ → C4_D4
                                {"id":"C4_D4","type":"chance","label":"C₄","weights":[0.3,0.35,0.35],"text":"d42: S'adapter a la culture chinoise","titre":"Prix des matieres premieres",
                                 "children":[
                                     {"id":"L8","type":"leaf","label":"-150 M","utility":-150,"text":""},
                                     {"id":"L9","type":"leaf","label":"+350 M","utility":350,"text":""},
                                     {"id":"L10","type":"leaf","label":"+750 M","utility":750,"text":""}]}
                            ]}
                       ]},
                      # ── child 1 of D1 = d₁₂ → C4_D0
                      {"id":"C4_D0","type":"chance","label":"C₄","weights":[0.6,0.4],"text":"","titre":"Prix des matieres premieres",
                       "children":[
                           {"id":"L11","type":"leaf","label":"-500 M","utility":-500,"text":""},
                           {"id":"L12","type":"leaf","label":"+250 M","utility":250,"text":""}]}
                  ]},
                 # ── child 1 of C2_D0 = D2
                 {"id":"D2","type":"decision","label":"D₂","text":"",
                  "children":[
                      # child 0 of D2 = d₂₁ → C3_D2
                      {"id":"C3_D2","type":"chance","label":"C₃","weights":[0.3,0.7],"text":"d21: Augmenter la marge fortement","titre":"Demande",
                       "children":[
                           {"id":"L13","type":"leaf","label":"+1 Md","utility":1000,"text":""},
                           {"id":"L14","type":"leaf","label":"-750 M","utility":-750,"text":""}]},
                      # child 1 of D2 = d₂₂ → C1_D2
                      {"id":"C1_D2","type":"chance","label":"C₁","weights":[0.6,0.4],"text":"d22: Augmenter la marge faiblement","titre":"Tension geopolitique",
                       "children":[
                           # child 0 of C1_D2 = D5
                           {"id":"D5","type":"decision","label":"D₅","text":"",
                            "children":[
                                # child 0 of D5 = d₅₁ → C3_D5
                                {"id":"C3_D5","type":"chance","label":"C₃","weights":[0.4,0.6],"text":"d51: Reduire la quantite en stock","titre":"Demande",
                                 "children":[
                                     {"id":"L15","type":"leaf","label":"+700 M","utility":700,"text":""},
                                     {"id":"L16","type":"leaf","label":"-400 M","utility":-400,"text":""}]},
                                # child 1 of D5 = d₅₂ → C4_D5
                                {"id":"C4_D5","type":"chance","label":"C₄","weights":[0.7,0.3],"text":"d52: Positionner sur le marche du low cost","titre":"Prix des matieres premieres",
                                 "children":[
                                     {"id":"L17","type":"leaf","label":"-1 Md","utility":-1000,"text":""},
                                     {"id":"L18","type":"leaf","label":"+2 Md","utility":2000,"text":""}]}
                            ]},
                           # child 1 of C1_D2 = D6
                           {"id":"D6","type":"decision","label":"D₆","text":"",
                            "children":[
                                # child 0 of D6 = d₆₁ → C3_D6
                                {"id":"C3_D6","type":"chance","label":"C₃","weights":[0.6,0.3,0.1],"text":"d61: Investir massivement sur le marketing","titre":"Demande",
                                 "children":[
                                     {"id":"L19","type":"leaf","label":"+1.3 Md","utility":1300,"text":""},
                                     {"id":"L20","type":"leaf","label":"+900 M","utility":900,"text":""},
                                     {"id":"L21","type":"leaf","label":"-100 M","utility":-100,"text":""}]},
                                # child 1 of D6 = d₆₂ → C2_D6
                                {"id":"C2_D6","type":"chance","label":"C₂","weights":[0.5,0.5],"text":"d62: Investir faiblement sur le marketing","titre":"Concurrence",
                                 "children":[
                                     {"id":"L22","type":"leaf","label":"-500 M","utility":-500,"text":""},
                                     {"id":"L23","type":"leaf","label":"+800 M","utility":800,"text":""}]}
                            ]}
                       ]}
                  ]}
             ]},
            # ── child 1 of D0 = d₀₂ → C3_D0
            {"id":"C3_D0","type":"chance","label":"C₃","weights":[0.01,0.19,0.8],"text":"d02: Ne pas agir et garder la meme strategie","titre":"Demande",
             "children":[
                 {"id":"L24","type":"leaf","label":"+200 M","utility":200,"text":""},
                 {"id":"L25","type":"leaf","label":"-2.5 Md","utility":-2500,"text":""},
                 {"id":"L26","type":"leaf","label":"-5 Md","utility":-5000,"text":""}]}
        ]
    },
    # "possibilistic": {
    #     "id":"D0","type":"decision","label":"D₀","text":"",
    #     "children":[
    #         {"id":"C2_D0","type":"chance","label":"C₂","weights":[1,0.4],"text":"d01: Changer de strategie","titre":"Concurrence",
    #          "children":[
    #              {"id":"D1","type":"decision","label":"D₁","text":"",
    #               "children":[
    #                   {"id":"C1_D1","type":"chance","label":"C₁","weights":[1,0.3],"text":"d11: Renforcer le positionnement en Chine","titre":"Tension geopolitique",
    #                    "children":[
    #                        {"id":"D3","type":"decision","label":"D₃","text":"",
    #                         "children":[
    #                             {"id":"C3_D3","type":"chance","label":"C₃","weights":[0.5,1],"text":"d31: Faire un partenariat avec une marque local","titre":"Demande",
    #                              "children":[
    #                                  {"id":"L1","type":"leaf","label":"0.786","utility":0.786,"text":""},
    #                                  {"id":"L2","type":"leaf","label":"0.514","utility":0.514,"text":""}]},
    #                             {"id":"C2_D3","type":"chance","label":"C₂","weights":[1,0.5],"text":"d32: Rebranding local","titre":"Concurrence",
    #                              "children":[
    #                                  {"id":"L3","type":"leaf","label":"0.6","utility":0.6,"text":""},
    #                                  {"id":"L4","type":"leaf","label":"0.814","utility":0.814,"text":""}]}
    #                         ]},
    #                        {"id":"D4","type":"decision","label":"D₄","text":"",
    #                         "children":[
    #                             {"id":"C3_D4","type":"chance","label":"C₃","weights":[1,0.4,0.2],"text":"d41: Diversifier la capacite de distribution","titre":"Demande",
    #                              "children":[
    #                                  {"id":"L5","type":"leaf","label":"0.786","utility":0.786,"text":""},
    #                                  {"id":"L6","type":"leaf","label":"0.743","utility":0.743,"text":""},
    #                                  {"id":"L7","type":"leaf","label":"0.713","utility":0.713,"text":""}]},
    #                             {"id":"C4_D4","type":"chance","label":"C₄","weights":[0.9,1,1],"text":"d42: S'adapter a la culture chinoise","titre":"Prix des matieres premieres",
    #                              "children":[
    #                                  {"id":"L8","type":"leaf","label":"0.693","utility":0.693,"text":""},
    #                                  {"id":"L9","type":"leaf","label":"0.764","utility":0.764,"text":""},
    #                                  {"id":"L10","type":"leaf","label":"0.821","utility":0.821,"text":""}]}
    #                         ]}
    #                    ]},
    #                   {"id":"C4_D0","type":"chance","label":"C₄","weights":[1,0.5],"text":"","titre":"Prix des matieres premieres",
    #                    "children":[
    #                        {"id":"L11","type":"leaf","label":"0.643","utility":0.643,"text":""},
    #                        {"id":"L12","type":"leaf","label":"0.750","utility":0.750,"text":""}]}
    #               ]},
    #              {"id":"D2","type":"decision","label":"D₂","text":"",
    #               "children":[
    #                   {"id":"C3_D2","type":"chance","label":"C₃","weights":[0.4,1],"text":"d21: Augmenter la marge fortement","titre":"Demande",
    #                    "children":[
    #                        {"id":"L13","type":"leaf","label":"0.857","utility":0.857,"text":""},
    #                        {"id":"L14","type":"leaf","label":"0.607","utility":0.607,"text":""}]},
    #                   {"id":"C1_D2","type":"chance","label":"C₁","weights":[1,0.5],"text":"d22: Augmenter la marge faiblement","titre":"Tension geopolitique",
    #                    "children":[
    #                        {"id":"D5","type":"decision","label":"D₅","text":"",
    #                         "children":[
    #                             {"id":"C3_D5","type":"chance","label":"C₃","weights":[0.7,1],"text":"d51: Reduire la quantite en stock","titre":"Demande",
    #                              "children":[
    #                                  {"id":"L15","type":"leaf","label":"0.814","utility":0.814,"text":""},
    #                                  {"id":"L16","type":"leaf","label":"0.657","utility":0.657,"text":""}]},
    #                             {"id":"C4_D5","type":"chance","label":"C₄","weights":[1,0.4],"text":"d52: Positionner sur le marche du low cost","titre":"Prix des matieres premieres",
    #                              "children":[
    #                                  {"id":"L17","type":"leaf","label":"0.571","utility":0.571,"text":""},
    #                                  {"id":"L18","type":"leaf","label":"1","utility":1,"text":""}]}
    #                         ]},
    #                        {"id":"D6","type":"decision","label":"D₆","text":"",
    #                         "children":[
    #                             {"id":"C3_D6","type":"chance","label":"C₃","weights":[1,0.6,0.4],"text":"d61: Investir massivement sur le marketing","titre":"Demande",
    #                              "children":[
    #                                  {"id":"L19","type":"leaf","label":"0.9","utility":0.9,"text":""},
    #                                  {"id":"L20","type":"leaf","label":"0.843","utility":0.843,"text":""},
    #                                  {"id":"L21","type":"leaf","label":"0.7","utility":0.7,"text":""}]},
    #                             {"id":"C2_D6","type":"chance","label":"C₂","weights":[1,1],"text":"d62: Investir faiblement sur le marketing","titre":"Concurrence",
    #                              "children":[
    #                                  {"id":"L22","type":"leaf","label":"0.643","utility":0.643,"text":""},
    #                                  {"id":"L23","type":"leaf","label":"0.829","utility":0.829,"text":""}]}
    #                         ]}
    #                    ]}
    #               ]}
    #          ]},
    #         {"id":"C3_D0","type":"chance","label":"C₃","weights":[0.1,0.3,1],"text":"d02: Ne pas agir et garder la meme strategie","titre":"Demande",
    #          "children":[
    #              {"id":"L24","type":"leaf","label":"0.743","utility":0.743,"text":""},
    #              {"id":"L25","type":"leaf","label":"0.357","utility":0.357,"text":""},
    #              {"id":"L26","type":"leaf","label":"0","utility":0,"text":""}]}
    #     ]
    # }
}

# ─────────────────────────────────────────────────────────────────
# 2.  DECISION-NODE → CHILD-INDEX MAP
#     d_ij  means  "decision node i, choice j"  → child index j-1
#     e.g. d₀₁ → D0 picks child 0  (C2_D0)
#          d₀₂ → D0 picks child 1  (C3_D0)
#          d₁₁ → D1 picks child 0  (C1_D1)
#          d₁₂ → D1 picks child 1  (C4_D0)
#     etc.
# ─────────────────────────────────────────────────────────────────
CHOICE_IDX = {
    "d₀₁": 0, "d₀₂": 1,
    "d₁₁": 0, "d₁₂": 1,
    "d₂₁": 0, "d₂₂": 1,
    "d₃₁": 0, "d₃₂": 1,
    "d₄₁": 0, "d₄₂": 1,
    "d₅₁": 0, "d₅₂": 1,
    "d₆₁": 0, "d₆₂": 1,
    "—":   None,          # node not reached in this strategy
}

# Map tree node-id → strategy key
NODE_STRAT_KEY = {
    "D0": "D0",
    "D1": "D1",
    "D2": "D2",
    "D3": "D3",
    "D4": "D4",
    "D5": "D5",
    "D6": "D6",
}

# ─────────────────────────────────────────────────────────────────
# 3.  STRATEGY DEFINITIONS  (26 strategies, as provided)
# ─────────────────────────────────────────────────────────────────
STRATEGY_DEFS = [
    {"name":"δ₁",  "D0":"d₀₁","D1":"d₁₂","D2":"d₂₁","D3":"—","D4":"—","D5":"—","D6":"—",
     "sets":"(D₀,C₂); (D₁,C₄); (D₂,C₃)"},
    {"name":"δ₂",  "D0":"d₀₁","D1":"d₁₂","D2":"d₂₂","D3":"—","D4":"—","D5":"d₅₁","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₃)"},
    {"name":"δ₃",  "D0":"d₀₁","D1":"d₁₂","D2":"d₂₂","D3":"—","D4":"—","D5":"d₅₁","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₂)"},
    {"name":"δ₄",  "D0":"d₀₁","D1":"d₁₂","D2":"d₂₂","D3":"—","D4":"—","D5":"d₅₂","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₃)"},
    {"name":"δ₅",  "D0":"d₀₁","D1":"d₁₂","D2":"d₂₂","D3":"—","D4":"—","D5":"d₅₂","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₂)"},
    {"name":"δ₆",  "D0":"d₀₁","D1":"d₁₁","D2":"d₂₁","D3":"d₃₁","D4":"d₄₁","D5":"—","D6":"—",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₃); (D₂,C₃)"},
    {"name":"δ₇",  "D0":"d₀₁","D1":"d₁₁","D2":"d₂₁","D3":"d₃₁","D4":"d₄₂","D5":"—","D6":"—",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₄); (D₂,C₃)"},
    {"name":"δ₈",  "D0":"d₀₁","D1":"d₁₁","D2":"d₂₁","D3":"d₃₂","D4":"d₄₁","D5":"—","D6":"—",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₃); (D₂,C₃)"},
    {"name":"δ₉",  "D0":"d₀₁","D1":"d₁₁","D2":"d₂₁","D3":"d₃₂","D4":"d₄₂","D5":"—","D6":"—",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₄); (D₂,C₃)"},
    {"name":"δ₁₀", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₁","D5":"d₅₁","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₃); (D₂,C₁); (D₅,C₃); (D₆,C₃)"},
    {"name":"δ₁₁", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₁","D5":"d₅₁","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₃); (D₂,C₁); (D₅,C₃); (D₆,C₂)"},
    {"name":"δ₁₂", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₁","D5":"d₅₂","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₃); (D₂,C₁); (D₅,C₄); (D₆,C₃)"},
    {"name":"δ₁₃", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₁","D5":"d₅₂","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₃); (D₂,C₁); (D₅,C₄); (D₆,C₂)"},
    {"name":"δ₁₄", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₂","D5":"d₅₁","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₃)"},
    {"name":"δ₁₅", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₂","D5":"d₅₁","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₂)"},
    {"name":"δ₁₆", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₂","D5":"d₅₂","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₃)"},
    {"name":"δ₁₇", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₁","D4":"d₄₂","D5":"d₅₂","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₃); (D₄,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₂)"},
    {"name":"δ₁₈", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₁","D5":"d₅₁","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₃); (D₂,C₁); (D₅,C₃); (D₆,C₃)"},
    {"name":"δ₁₉", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₁","D5":"d₅₁","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₃); (D₂,C₁); (D₅,C₃); (D₆,C₂)"},
    {"name":"δ₂₀", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₁","D5":"d₅₂","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₃); (D₂,C₁); (D₅,C₄); (D₆,C₃)"},
    {"name":"δ₂₁", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₁","D5":"d₅₂","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₃); (D₂,C₁); (D₅,C₄); (D₆,C₂)"},
    {"name":"δ₂₂", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₂","D5":"d₅₁","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₃)"},
    {"name":"δ₂₃", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₂","D5":"d₅₁","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₄); (D₂,C₁); (D₅,C₃); (D₆,C₂)"},
    {"name":"δ₂₄", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₂","D5":"d₅₂","D6":"d₆₁",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₃)"},
    {"name":"δ₂₅", "D0":"d₀₁","D1":"d₁₁","D2":"d₂₂","D3":"d₃₂","D4":"d₄₂","D5":"d₅₂","D6":"d₆₂",
     "sets":"(D₀,C₂); (D₁,C₁); (D₃,C₂); (D₄,C₄); (D₂,C₁); (D₅,C₄); (D₆,C₂)"},
    {"name":"δ₂₆", "D0":"d₀₂","D1":"—","D2":"—","D3":"—","D4":"—","D5":"—","D6":"—",
     "sets":"(D₀,C₃)"},
]
STRAT_NAMES = [s["name"] for s in STRATEGY_DEFS]
N_STRAT     = len(STRATEGY_DEFS)

PAL = [
    "#5b9bd5","#d4a042","#4aaf7a","#8a7af0","#d46050",
    "#3dbdb0","#e86bb5","#9acd50","#d4849a","#7ab0d5",
    "#c4703a","#6a9fd5","#b0d46a","#d56aa0","#7af0c0",
    "#d5b05b","#5ba8d5","#d0a0d5","#a0d598","#d5a080",
    "#60b0d5","#d5d060","#8070d5","#d58060","#70d5a0",
    "#d07070",
]

# ─────────────────────────────────────────────────────────────────
# 4.  MODE-AWARE CRITERIA
# ─────────────────────────────────────────────────────────────────
CRITERIA_BY_MODE = {
    "probabilistic": ["EU linear", "EU concave", "EU convex"],
    # "possibilistic":  ["U_opt", "U_pes", "PU"],
}

# ─────────────────────────────────────────────────────────────────
# 5.  BACKWARD INDUCTION  (full tree, unconstrained)
# ─────────────────────────────────────────────────────────────────
def solve_node(node, crit):
    """Standard backward induction — decision nodes pick the max child."""
    if node["type"] == "leaf":
        return node["utility"], -1, []
    child_results = [solve_node(c, crit) for c in node["children"]]
    cv = [r[0] for r in child_results]
    w  = node.get("weights", [])
    if node["type"] == "decision":
        val = max(cv)
        opt_child = cv.index(val)
    else:
        val, opt_child = _chance_val(cv, w, crit), -1
    return val, opt_child, child_results


def _chance_val(cv, w, crit):
    """
    Calculates the expected utility based on the specified risk profile.
    cv: list of values (consequences)
    w:  list of weights (probabilities)
    crit: string defining the utility function
    """
    if crit == "EU linear":
        # Linear: u(x) = x
        return sum(wi * v for wi, v in zip(w, cv))
        
    elif crit == "EU concave":
        # Concave: u(x) = sqrt(6000 + x)
        # Using max(..., 0) to prevent domain errors with sqrt
        return sum(wi * math.sqrt(max(6000 + v, 0)) for wi, v in zip(w, cv))
        
    elif crit == "EU convex":
        # Convex: u(x) = (6000 + x)^2
        return sum(wi * ((6000 + v)**2) for wi, v in zip(w, cv))
        
    else:
        # Default to linear if criterion is unrecognized
        return sum(wi * v for wi, v in zip(w, cv))

def collect_opt_path(node, result, path=None):
    if path is None: path = set()
    val, opt_child, child_results = result
    path.add(node["id"])
    if not child_results: return path
    if node["type"] == "decision" and opt_child >= 0:
        collect_opt_path(node["children"][opt_child], child_results[opt_child], path)
    else:
        for child, cr in zip(node["children"], child_results):
            collect_opt_path(child, cr, path)
    return path


# ─────────────────────────────────────────────────────────────────
# 6.  STRATEGY EVALUATION  (fixed decision choices per strategy)
#     Each decision node in the tree is forced to the child index
#     specified by the strategy's dXX key.  Unreachable branches
#     are still evaluated structurally but their decision is forced;
#     this naturally prunes the computation.
# ─────────────────────────────────────────────────────────────────
def eval_strategy(tree, crit, strat_def):
    """
    Evaluate the expected value of a fixed pure strategy.
    Decision nodes pick the child index mandated by the strategy.
    Chance nodes aggregate using `crit`.
    """
    # Build a lookup: node-id → forced child index (or None if not reached)
    forced = {}
    for key, val in strat_def.items():
        if key in NODE_STRAT_KEY and val != "—" and val is not None:
            idx = CHOICE_IDX.get(val)
            if idx is not None:
                node_id = key   # "D0","D1",... map directly to node ids
                forced[node_id] = idx

    def ev(node):
        if node["type"] == "leaf":
            return node["utility"]
        nid = node["id"]
        cv  = [ev(c) for c in node["children"]]
        w   = node.get("weights", [])
        if node["type"] == "decision":
            # Use forced choice if available, else pick max (shouldn't happen
            # for a complete strategy, but safe fallback)
            idx = forced.get(nid)
            if idx is not None and idx < len(cv):
                return cv[idx]
            return max(cv)
        # chance node
        return _chance_val(cv, w, crit)

    return ev(tree)


def fmt(v):
    if not isinstance(v, (int, float)): return "—"
    if abs(v) < 2: return f"{v:.4f}"
    return ("+" if v >= 0 else "") + f"{round(v):,}" + " M"

def fmt_short(v):
    if not isinstance(v, (int, float)): return "—"
    if abs(v) < 2: return f"{v:.3f}"
    return ("+" if v >= 0 else "") + f"{round(v):,}"


# ─────────────────────────────────────────────────────────────────
# 7.  AGGREGATE SCORE HELPERS
# ─────────────────────────────────────────────────────────────────
def scores_for_crit(tree, crit):
    """Return list of per-strategy scores (length = N_STRAT)."""
    return [eval_strategy(tree, crit, s) for s in STRATEGY_DEFS]


def compute_wins(tree, crits):
    """Number of criteria (out of len(crits)) each strategy wins."""
    all_scores = {c: scores_for_crit(tree, c) for c in crits}
    wins = []
    for i in range(N_STRAT):
        w = sum(1 for c in crits if all_scores[c][i] == max(all_scores[c]))
        wins.append(w)
    return wins


def norm_arr(arr):
    mn, mx = min(arr), max(arr)
    if mx == mn: return [0.5]*len(arr)
    return [(v-mn)/(mx-mn) for v in arr]


# ─────────────────────────────────────────────────────────────────
# 8.  GRAPHVIZ TREE
# ─────────────────────────────────────────────────────────────────
BG    = "#FFFFFF"
BG2   = "#FFFFFF"
GOLD  = "#f0c040"
GRID_C = "rgba(255,255,255,0.05)"
TICK_C = "#7a8799"

def hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color + alpha [0-1] to an rgba() string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def build_graphviz(tree_data, crit, mode):
    result = solve_node(tree_data, crit)
    opt    = collect_opt_path(tree_data, result)
    pfx    = "p=" if mode == "probabilistic" else "π="

    dot = graphviz.Digraph(graph_attr={
        "rankdir":"LR","bgcolor":BG,
        "splines":"curved","nodesep":"0.3","ranksep":"0.9",
    })

    def add(n, parent=None, elbl=""):
        nid = n["id"]
        on  = nid in opt

        if n["type"] == "decision":
            dot.node(nid, n["label"], shape="square", style="filled",
                     fillcolor="#d4a042" if on else "#3a2a0a",
                     fontcolor="#1a0f00" if on else "#d4a042",
                     color=GOLD if on else "#8a6020",
                     penwidth="2.5" if on else "1",
                     fontname="Courier", fontsize="11")
        elif n["type"] == "chance":
            dot.node(nid, n["label"], shape="circle", style="filled",
                     fillcolor="#1e4a7a" if on else "#0e1e30",
                     fontcolor="#a0d0ff" if on else "#5b9bd5",
                     color=GOLD if on else "#2d5a8a",
                     penwidth="2.5" if on else "1",
                     fontname="Courier", fontsize="11",
                     tooltip=n["titre"])
        else:
            v   = n["utility"]
            pos = v >= 0
            dot.node(nid, f"{n['label']}\n({v})", shape="box",
                     style="filled,rounded",
                     fillcolor="#0e2a1c" if pos else "#2a1510",
                     fontcolor=GOLD if on else ("#4aaf7a" if pos else "#d46050"),
                     color=GOLD if on else ("#4aaf7a55" if pos else "#d4605055"),
                     penwidth="2" if on else "0.5",
                     fontname="Courier", fontsize="9")

        if parent:
            eon = parent in opt and nid in opt
            dot.edge(parent, nid, label=elbl,
                     color=GOLD if eon else "#2a3347",
                     penwidth="2.5" if eon else "0.8",
                     fontcolor=GOLD if eon else "#3d4d63",
                     fontname="Courier", fontsize="8",
                     tooltip=n["text"])

        for i, ch in enumerate(n.get("children", [])):
            wi  = n.get("weights", [None]*(i+1))[i] if i < len(n.get("weights",[])) else None
            lbl = f"{pfx}{wi}" if wi is not None else ""
            add(ch, nid, lbl)

    add(tree_data)
    return dot


# ─────────────────────────────────────────────────────────────────
# 9.  PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────
def _layout(fig, h=300, **kw):
    fig.update_layout(
        height=h, plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=50,r=20,t=10,b=50),
        font=dict(color=TICK_C,family="monospace"),
        **kw)


def bar_chart(tree, crits):
    fig = go.Figure()
    for i, c in enumerate(crits):
        sc = scores_for_crit(tree, c)
        fig.add_trace(go.Bar(
            name=c, x=STRAT_NAMES, y=sc,
            marker_color=hex_alpha(PAL[i%len(PAL)], 0.16),
            marker_line_color=PAL[i%len(PAL)], marker_line_width=1.5,
            marker_cornerradius=3))
    _layout(fig, h=340, barmode="group",
            xaxis=dict(tickfont=dict(size=9), gridcolor=GRID_C, tickangle=-45),
            yaxis=dict(tickfont=dict(size=9), gridcolor=GRID_C),
            legend=dict(orientation="h", y=-0.35, font_size=10))
    return fig


def radar_chart(tree, crits):
    """Top-5 strategies by active criterion, radar over all crits."""
    sc0  = scores_for_crit(tree, crits[0])
    top5 = sorted(range(N_STRAT), key=lambda i: sc0[i], reverse=True)[:5]
    cats = crits + [crits[0]]
    fig  = go.Figure()
    for rank, i in enumerate(top5):
        vals = [norm_arr(scores_for_crit(tree, c))[i] for c in crits]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=STRAT_NAMES[i],
            fill="toself", opacity=0.5,
            line_color=PAL[i%len(PAL)],
            fillcolor=hex_alpha(PAL[i%len(PAL)], 0.16)))
    fig.update_layout(
        height=340, paper_bgcolor=BG,
        margin=dict(l=30,r=30,t=20,b=50),
        polar=dict(bgcolor=BG2,
            radialaxis=dict(visible=True, range=[0,1],
                tickfont=dict(size=8,color=TICK_C), gridcolor="rgba(255,255,255,.07)"),
            angularaxis=dict(tickfont=dict(size=9,color=TICK_C,family="monospace"))),
        legend=dict(orientation="h", y=-0.2, font_size=10, font_color=TICK_C))
    return fig


def heatmap_chart(tree, crits):
    all_sc   = {c: scores_for_crit(tree, c) for c in crits}
    all_norm = {c: norm_arr(all_sc[c])       for c in crits}
    z    = [[all_norm[c][i] for c in crits] for i in range(N_STRAT)]
    text = [[fmt_short(all_sc[c][i]) for c in crits] for i in range(N_STRAT)]
    fig  = go.Figure(go.Heatmap(
        z=z, x=crits, y=STRAT_NAMES,
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#2a1510"],[0.5,"#3a2a0a"],[1,"#0e2a1c"]],
        showscale=True,
        colorbar=dict(
            title="Norm.", tickfont=dict(size=9,color=TICK_C),
            titlefont=dict(size=9,color=TICK_C), len=0.8)))
    fig.update_layout(
        height=max(320, N_STRAT*14+80),
        plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(l=55,r=90,t=10,b=40),
        xaxis=dict(tickfont=dict(size=10,color=TICK_C,family="monospace")),
        yaxis=dict(tickfont=dict(size=9,color=TICK_C,family="monospace"),
                   autorange="reversed"))
    return fig


def wins_chart(wins, crits):
    colors = [PAL[i%len(PAL)] for i in range(N_STRAT)]
    fig = go.Figure(go.Bar(
        x=wins, y=STRAT_NAMES, orientation="h",
        marker_color=[hex_alpha(c, 0.27) for c in colors],
        marker_line_color=colors, marker_line_width=1.5,
        marker_cornerradius=3,
        text=wins, textposition="outside",
        textfont=dict(color=TICK_C, size=9, family="monospace")))
    _layout(fig, h=max(280, N_STRAT*14+60),
            xaxis=dict(title="Criteria won", range=[0, len(crits)+.5],
                       tickfont=dict(size=9), gridcolor=GRID_C,
                       titlefont=dict(color=TICK_C)),
            yaxis=dict(tickfont=dict(size=9), autorange="reversed"))
    return fig


def scatter_chart(tree, crits, active_crit):
    sc_active = scores_for_crit(tree, active_crit)
    all_sc    = [scores_for_crit(tree, c) for c in crits]
    spread    = [max(s[i] for s in all_sc) - min(s[i] for s in all_sc)
                 for i in range(N_STRAT)]
    fig = go.Figure()
    for i, (s, x, y) in enumerate(zip(STRAT_NAMES, spread, sc_active)):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            name=s, text=[s], textposition="top center",
            textfont=dict(color=PAL[i%len(PAL)], size=9, family="monospace"),
            marker=dict(size=12, color=PAL[i%len(PAL)],
                        line=dict(width=1.5, color=BG))))
    _layout(fig, h=320, showlegend=False,
            xaxis=dict(title="Criterion spread",
                       tickfont=dict(size=9), gridcolor=GRID_C,
                       titlefont=dict(color=TICK_C, size=10)),
            yaxis=dict(title=active_crit,
                       tickfont=dict(size=9), gridcolor=GRID_C,
                       titlefont=dict(color=TICK_C, size=10)))
    return fig


# ─────────────────────────────────────────────────────────────────
# 10. STREAMLIT UI
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Decision Tree Solver", layout="wide", page_icon="🌳")

st.markdown(f"""<style>
body,.stApp{{background:{BG};color:grey;}}
[data-testid="stSidebar"]{{background:{BG2}!important;border-right:1px solid #2a3347;}}
[data-testid="stSidebar"] *{{color:#7a8799!important;}}
.stTabs [data-baseweb="tab-list"]{{background:{BG2};border-radius:8px;gap:3px;padding:4px;}}
.stTabs [data-baseweb="tab"]{{background:none;border-radius:6px;color:#7a8799;
  font-family:monospace;font-size:12px;}}
.stTabs [aria-selected="true"]{{background:#26273033!important;color:#5b9bd5!important;}}
h1,h2,h3{{color:#9cb6e6!important;font-family:monospace!important;}}
.stDataFrame{{border-radius:8px;overflow:hidden;}}
div[data-testid="metric-container"]{{background:{BG2};border:1px solid #2a3347;
  border-radius:8px;padding:12px 14px;}}
.stRadio>div{{gap:6px;}}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Decision Tree Solver")
    st.divider()

    mode  = st.radio("Tree mode", ["probabilistic"], horizontal=True)
    crits = CRITERIA_BY_MODE[mode]
    tree  = TREES[mode]
    st.divider()

    crit      = st.radio("Criterion", crits)
    st.divider()

    # optimal result card
    opt_val, _, _ = solve_node(tree, crit)
    if opt_val >= 0:
        st.success(f"**DP optimum:** {fmt(opt_val)}\n\n*{crit} · {mode}*")
    else:
        st.error(f"**DP optimum:** {fmt(opt_val)}\n\n*{crit} · {mode}*")

# ── Tabs ──────────────────────────────────────────────────────────
tab_tree, tab_strat, tab_dash= st.tabs(
    [" Decision Tree", " Strategies", " Dashboard"])

# ─── DASHBOARD ────────────────────────────────────────────────────
with tab_dash:
    scores  = scores_for_crit(tree, crit)
    best_v  = max(scores);  best_i = scores.index(best_v)
    wins    = compute_wins(tree, crits)
    rob_i   = wins.index(max(wins))
    std_v   = float(np.std(scores))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal value (DP)",   fmt(opt_val),         "unconstrained")
    c2.metric("Best strategy value",  fmt(best_v),          STRAT_NAMES[best_i])
    c3.metric("Most robust strategy", STRAT_NAMES[rob_i],   f"{max(wins)}/{len(crits)} criteria")
    c4.metric("Score std dev",        f"{std_v:.3f}",       "across strategies")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Scores by criterion")
        st.plotly_chart(bar_chart(tree, crits), use_container_width=True, key="dash_bar")
    with col2:
        st.markdown("### Radar — top-5 strategies (normalized)")
        st.plotly_chart(radar_chart(tree, crits), use_container_width=True, key="dash_radar")

    st.markdown("### Cross-criteria heatmap — all strategies × all criteria")
    st.plotly_chart(heatmap_chart(tree, crits), use_container_width=True, key="dash_heatmap")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Robustness — criteria won per strategy")
        st.plotly_chart(wins_chart(wins, crits), use_container_width=True, key="dash_wins")
    with col4:
        st.markdown("### Criterion spread vs. score")
        st.plotly_chart(scatter_chart(tree, crits, crit), use_container_width=True, key="dash_scatter")

# ─── DECISION TREE ────────────────────────────────────────────────
with tab_tree:
    st.markdown(f"### Decision Tree — **{mode}** · path criterion: **{crit}**")

    dot = build_graphviz(tree, crit, mode)
    st.graphviz_chart(dot, use_container_width=True)

    res = solve_node(tree, crit)
    opt = collect_opt_path(tree, res)
    non_leaf = sorted(n for n in opt if not n.startswith("L"))
    ca, cb = st.columns(2)
    ca.metric("Optimal DP value", fmt(res[0]), crit)
    cb.info(f"**Optimal path:** {' → '.join(non_leaf)}")

# ─── STRATEGIES ───────────────────────────────────────────────────
with tab_strat:
    st.markdown(f"### Strategy Set Δ — *{mode}* · criterion: *{crit}*")
    st.caption(f"AD = {N_STRAT} pure strategies · fixed decision-node choices · "
               "scores computed by constrained backward induction")

    wins     = compute_wins(tree, crits)
    best_win = max(wins)
    all_sc   = {c: scores_for_crit(tree, c) for c in crits}

    # ── Strategy matrix ───────────────────────────────────────────
    st.markdown("#### Strategy matrix")
    SHOW_COLS = ["D₀","D₁","D₂","D₃","D₄","D₅","D₆"]
    SHOW_KEYS = ["D0","D1","D2","D3","D4","D5","D6"]
    mat_rows = []
    for i, s in enumerate(STRATEGY_DEFS):
        row = {"Δ": s["name"]}
        for col, key in zip(SHOW_COLS, SHOW_KEYS):
            row[col] = s.get(key, "—")
        row["Sets"] = s["sets"]
        row["Won"]  = f"{wins[i]}/{len(crits)}"
        row["★"]    = "✓" if wins[i] == best_win else ""
        mat_rows.append(row)

    df_mat = pd.DataFrame(mat_rows).set_index("Δ")

    # Highlight optimal rows
    def highlight_opt(row):
        if row["★"] == "✓":
            return [
                "background-color:#1e1840;color:#8a7af0;font-weight:600"
            ] * len(row)
        return [""] * len(row)

    styled_mat = (df_mat.style
                  .apply(highlight_opt, axis=1)
                  .set_properties(**{"font-size":"11px","font-family":"monospace"}))
    st.dataframe(styled_mat, use_container_width=True, height=min(600, N_STRAT*36+50))

    st.divider()

    # ── Evaluation table ──────────────────────────────────────────
    st.markdown("#### Evaluation table — score per strategy per criterion")
    eval_rows = []
    for i, s in enumerate(STRATEGY_DEFS):
        row = {"Strategy": s["name"]}
        for c in crits:
            row[c] = all_sc[c][i]
        row["Spread"] = max(all_sc[c][i] for c in crits) - min(all_sc[c][i] for c in crits)
        row["Criteria won"] = wins[i]
        row["Optimal"] = "✓" if wins[i] == best_win else ""
        eval_rows.append(row)

    df_eval = pd.DataFrame(eval_rows).set_index("Strategy")

    def color_score(val):
        if not isinstance(val, float): return ""
        if val > 0:   return "color:#4aaf7a;font-weight:500"
        elif val < 0: return "color:#d46050"
        return ""

    def highlight_opt_eval(row):
        if row["Optimal"] == "✓":
            return ["background-color:#1e1840;color:#8a7af0;font-weight:600"] * len(row)
        return [""] * len(row)

    fmt_cols = {c: lambda v, _c=c: fmt_short(v) for c in crits}
    fmt_cols["Spread"] = lambda v: fmt_short(v)

    styled_eval = (df_eval.style
                   .apply(highlight_opt_eval, axis=1)
                   .applymap(color_score, subset=crits)
                   .format(fmt_cols)
                   .highlight_max(subset=crits, color="#0e2a1c")
                   .highlight_min(subset=crits, color="#2a1510")
                   .set_properties(**{"font-size":"11px","font-family":"monospace"}))
    st.dataframe(styled_eval, use_container_width=True, height=min(600, N_STRAT*36+50))

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Robustness — criteria won")
        st.plotly_chart(wins_chart(wins, crits), use_container_width=True, key="strat_wins")
    with col_b:
        st.markdown("#### Cross-criteria heatmap")
        st.plotly_chart(heatmap_chart(tree, crits), use_container_width=True, key="strat_heatmap")