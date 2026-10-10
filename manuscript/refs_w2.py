# -*- coding: utf-8 -*-
"""Tài liệu tham khảo của bản thảo W2 (kiểu Elsevier đánh số, IJAR), đánh số theo thứ tự trích.

Mỗi mục: key -> (chuỗi trích dẫn, DOI hoặc None, "CR"|"DC"|None = nguồn đối chiếu metadata).
Metadata đối chiếu Crossref ngày 25/09/2026 (xem W2_references_verified.md);
build_w2.py --check-doi kiểm lại DOI, năm và họ tác giả đầu.
"""

REFS = {
    # --- hệ mờ đơn điệu -------------------------------------------------------------------------
    "mamdani1975": ("E.H. Mamdani, S. Assilian, An experiment in linguistic synthesis with a fuzzy logic controller, Int. J. Man-Mach. Stud. 7 (1) (1975) 1–13.", "10.1016/S0020-7373(75)80002-2", "CR"),
    "ruspini1969": ("E.H. Ruspini, A new approach to clustering, Inf. Control 15 (1) (1969) 22–32.", "10.1016/S0019-9958(69)90591-9", "CR"),
    "kosko1997": ("B. Kosko, Fuzzy Engineering, Prentice Hall, Upper Saddle River, NJ, 1997.", None, None),
    "vbdb2006": ("E. Van Broekhoven, B. De Baets, Fast and accurate center of gravity defuzzification of fuzzy system outputs defined on trapezoidal fuzzy partitions, Fuzzy Sets Syst. 157 (7) (2006) 904–918.", "10.1016/j.fss.2005.11.005", "CR"),
    "vbdb2008": ("E. Van Broekhoven, B. De Baets, Monotone Mamdani–Assilian models under mean of maxima defuzzification, Fuzzy Sets Syst. 159 (21) (2008) 2819–2844.", "10.1016/j.fss.2008.03.014", "CR"),
    "vbdb2009": ("E. Van Broekhoven, B. De Baets, Only smooth rule bases can generate monotone Mamdani–Assilian models under center-of-gravity defuzzification, IEEE Trans. Fuzzy Syst. 17 (5) (2009) 1157–1174.", "10.1109/TFUZZ.2009.2023328", "CR"),
    "won2002": ("J.M. Won, S.Y. Park, J.S. Lee, Parameter conditions for monotonic Takagi–Sugeno–Kang fuzzy system, Fuzzy Sets Syst. 132 (2) (2002) 135–146.", "10.1016/S0165-0114(02)00121-5", "CR"),
    "kouikoglou2009": ("V.S. Kouikoglou, Y.A. Phillis, On the monotonicity of hierarchical sum–product fuzzy systems, Fuzzy Sets Syst. 160 (24) (2009) 3530–3538.", "10.1016/j.fss.2009.02.001", "CR"),
    "tay2011": ("K.M. Tay, C.P. Lim, On monotonic sufficient conditions of fuzzy inference systems and their applications, Int. J. Uncertain. Fuzziness Knowl.-Based Syst. 19 (5) (2011) 731–757.", "10.1142/S0218488511007210", "CR"),
    "seki2010": ("H. Seki, H. Ishii, M. Mizumoto, On the monotonicity of fuzzy-inference methods related to T–S inference method, IEEE Trans. Fuzzy Syst. 18 (3) (2010) 629–634.", "10.1109/TFUZZ.2010.2046668", "CR"),
    "lindskog2000": ("P. Lindskog, L. Ljung, Ensuring monotonic gain characteristics in estimated models by fuzzy model structures, Automatica 36 (2) (2000) 311–317.", "10.1016/S0005-1098(99)00154-5", "CR"),
    "pang2016": ("L.M. Pang, K.M. Tay, C.P. Lim, Monotone fuzzy rule relabeling for the zero-order TSK fuzzy inference system, IEEE Trans. Fuzzy Syst. 24 (6) (2016) 1455–1463.", "10.1109/TFUZZ.2016.2540059", "CR"),
    "kerk2025": ("Y.W. Kerk, C.H. Jong, W.L. Chang, C.J. Tan, K.M. Tay, C.P. Lim, Designing monotone Takagi–Sugeno–Kang fuzzy inference systems with new joint sufficient conditions, Fuzzy Sets Syst. 502 (2025) 109217.", "10.1016/j.fss.2024.109217", "CR"),
    # --- Choquet, độ đo mờ ---------------------------------------------------------------------
    "choquet1954": ("G. Choquet, Theory of capacities, Ann. Inst. Fourier 5 (1954) 131–295.", "10.5802/aif.53", "CR"),
    "grabisch1997": ("M. Grabisch, k-order additive discrete fuzzy measures and their representation, Fuzzy Sets Syst. 92 (2) (1997) 167–189.", "10.1016/S0165-0114(97)00168-1", "CR"),
    "chateauneuf1989": ("A. Chateauneuf, J.-Y. Jaffray, Some characterizations of lower probabilities and other monotone capacities through the use of Möbius inversion, Math. Soc. Sci. 17 (3) (1989) 263–283.", "10.1016/0165-4896(89)90056-5", "CR"),
    "miranda1999": ("P. Miranda, M. Grabisch, Optimization issues for fuzzy measures, Int. J. Uncertain. Fuzziness Knowl.-Based Syst. 7 (6) (1999) 545–560.", "10.1142/S0218488599000477", "CR"),
    "grabisch2008": ("M. Grabisch, I. Kojadinovic, P. Meyer, A review of methods for capacity identification in Choquet integral based multi-attribute utility theory: Applications of the Kappalab R package, Eur. J. Oper. Res. 186 (2) (2008) 766–785.", "10.1016/j.ejor.2007.02.025", "CR"),
    "grabisch2010": ("M. Grabisch, C. Labreuche, A decade of application of the Choquet and Sugeno integrals in multi-criteria decision aid, Ann. Oper. Res. 175 (1) (2010) 247–286.", "10.1007/s10479-009-0655-8", "CR"),
    "beliakov2009": ("G. Beliakov, Construction of aggregation functions from data using linear programming, Fuzzy Sets Syst. 160 (1) (2009) 65–75.", "10.1016/j.fss.2008.07.004", "CR"),
    "beliakov2019": ("G. Beliakov, J.-Z. Wu, Learning fuzzy measures from data: Simplifications and optimisation strategies, Inf. Sci. 494 (2019) 100–113.", "10.1016/j.ins.2019.04.042", "CR"),
    "kojadinovic2007": ("I. Kojadinovic, Minimum variance capacity identification, Eur. J. Oper. Res. 177 (1) (2007) 498–514.", "10.1016/j.ejor.2005.10.059", "CR"),
    "tehrani2011": ("A. Fallah Tehrani, W. Cheng, E. Hüllermeier, Choquistic regression: Generalizing logistic regression using the Choquet integral, in: Proc. 7th Conf. Eur. Soc. Fuzzy Logic Technol. (EUSFLAT-2011), Atlantis Press, 2011.", "10.2991/eusflat.2011.86", "CR"),
    "tehrani2012": ("A. Fallah Tehrani, W. Cheng, K. Dembczyński, E. Hüllermeier, Learning monotone nonlinear models using the Choquet integral, Mach. Learn. 89 (1–2) (2012) 183–211.", "10.1007/s10994-012-5318-3", "CR"),
    "tehrani2013": ("A. Fallah Tehrani, E. Hüllermeier, Ordinal choquistic regression, in: Proc. 8th Conf. Eur. Soc. Fuzzy Logic Technol. (EUSFLAT-2013), Atlantis Press, 2013.", "10.2991/eusflat.2013.119", "CR"),
    "hullermeier2013": ("E. Hüllermeier, A. Fallah Tehrani, Efficient learning of classifiers based on the 2-additive Choquet integral, in: C. Moewes, A. Nürnberger (Eds.), Computational Intelligence in Intelligent Data Analysis, Stud. Comput. Intell. 445, Springer, Berlin, Heidelberg, 2013, pp. 17–29.", "10.1007/978-3-642-32378-2_2", "CR"),
    "bresson2020": ("R. Bresson, J. Cohen, E. Hüllermeier, C. Labreuche, M. Sebag, Neural representation and learning of hierarchical 2-additive Choquet integrals, in: Proc. 29th Int. Joint Conf. Artif. Intell. (IJCAI-20), 2020, pp. 1984–1991.", "10.24963/ijcai.2020/275", "CR"),
    "islam2020": ("M.A. Islam, D.T. Anderson, A.J. Pinar, T.C. Havens, G. Scott, J.M. Keller, Enabling explainable fusion in deep learning with fuzzy integral neural networks, IEEE Trans. Fuzzy Syst. 28 (7) (2020) 1291–1300.", "10.1109/TFUZZ.2019.2917124", "CR"),
    # --- học máy đơn điệu, tối ưu, thống kê ----------------------------------------------------
    "cano2019": ("J.-R. Cano, P.A. Gutiérrez, B. Krawczyk, M. Woźniak, S. García, Monotonic classification: An overview on algorithms, performance measures and data sets, Neurocomputing 341 (2019) 168–182.", "10.1016/j.neucom.2019.02.024", "CR"),
    "ke2017": ("G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, LightGBM: A highly efficient gradient boosting decision tree, in: Adv. Neural Inf. Process. Syst. 30 (NIPS 2017), 2017.", None, None),
    "gupta2016": ("M. Gupta, A. Cotter, J. Pfeifer, K. Voevodski, K. Canini, A. Mangylov, W. Moczydlowski, A. van Esbroeck, Monotonic calibrated interpolated look-up tables, J. Mach. Learn. Res. 17 (109) (2016) 1–47.", None, None),
    "you2017": ("S. You, D. Ding, K. Canini, J. Pfeifer, M. Gupta, Deep lattice networks and partial monotonic functions, in: Adv. Neural Inf. Process. Syst. 30 (NIPS 2017), 2017.", None, None),
    "byrd1995": ("R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, A limited memory algorithm for bound constrained optimization, SIAM J. Sci. Comput. 16 (5) (1995) 1190–1208.", "10.1137/0916069", "CR"),
    "zhu1997": ("C. Zhu, R.H. Byrd, P. Lu, J. Nocedal, Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization, ACM Trans. Math. Softw. 23 (4) (1997) 550–560.", "10.1145/279232.279236", "CR"),
    "byrd1999": ("R.H. Byrd, M.E. Hribar, J. Nocedal, An interior point algorithm for large-scale nonlinear programming, SIAM J. Optim. 9 (4) (1999) 877–900.", "10.1137/S1052623497325107", "CR"),
    "kraft1988": ("D. Kraft, A software package for sequential quadratic programming, Tech. Rep. DFVLR-FB 88-28, DLR German Aerospace Center — Institute for Flight Mechanics, Köln, Germany, 1988.", None, None),
    "nadeau2003": ("C. Nadeau, Y. Bengio, Inference for the generalization error, Mach. Learn. 52 (3) (2003) 239–281.", "10.1023/A:1024068626366", "CR"),
    "benjamini1995": ("Y. Benjamini, Y. Hochberg, Controlling the false discovery rate: A practical and powerful approach to multiple testing, J. R. Stat. Soc. Ser. B (Methodol.) 57 (1) (1995) 289–300.", "10.1111/j.2517-6161.1995.tb02031.x", "CR"),
    # --- dữ liệu ---------------------------------------------------------------------------------
    "little2009": ("M.A. Little, P.E. McSharry, E.J. Hunter, J. Spielman, L.O. Ramig, Suitability of dysphonia measurements for telemonitoring of Parkinson's disease, IEEE Trans. Biomed. Eng. 56 (4) (2009) 1015–1022.", "10.1109/TBME.2008.2005954", "CR"),
    "wolberg1990": ("W.H. Wolberg, O.L. Mangasarian, Multisurface method of pattern separation for medical diagnosis applied to breast cytology, Proc. Natl. Acad. Sci. USA 87 (23) (1990) 9193–9196.", "10.1073/pnas.87.23.9193", "CR"),
    "detrano1989": ("R. Detrano, A. Janosi, W. Steinbrunn, M. Pfisterer, J.-J. Schmid, S. Sandhu, K.H. Guppy, S. Lee, V. Froelicher, International application of a new probability algorithm for the diagnosis of coronary artery disease, Am. J. Cardiol. 64 (5) (1989) 304–310.", "10.1016/0002-9149(89)90524-9", "CR"),
    "allison2008": ("C. Allison, S. Baron-Cohen, S. Wheelwright, T. Charman, J. Richler, G. Pasco, C. Brayne, The Q-CHAT (Quantitative CHecklist for Autism in Toddlers): A normally distributed quantitative measure of autistic traits at 18–24 months of age: Preliminary report, J. Autism Dev. Disord. 38 (8) (2008) 1414–1425.", "10.1007/s10803-007-0509-7", "CR"),
    "niedzwiecka2022": ("A. Niedźwiecka, E. Pisula, Symptoms of autism spectrum disorders measured by the Qualitative Checklist for Autism in Toddlers in a large sample of Polish toddlers, Int. J. Environ. Res. Public Health 19 (5) (2022) 3072.", "10.3390/ijerph19053072", "CR"),
    "niedzwiecka_data": ("A. Niedźwiecka, Q-CHAT scores of Polish toddlers with autism spectrum disorders and typically developing controls [dataset], Mendeley Data, version 2, 2020.", "10.17632/tmpkt2mfkg.2", "DC"),
}
