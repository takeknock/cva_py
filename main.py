import numpy as np
import matplotlib.pyplot as plt
import QuantLib as Qlb

# https://github.com/mgroncki/IPythonScripts/blob/master/CVA_calculation_I.ipynb

def make_swap(start, maturity, notional, fixed_rate, index, typ=Qlb.VanillaSwap.Payer):
    end = Qlb.TARGET().advance(start, maturity)
    fixed_leg_tenor = Qlb.Period('6m')
    fixed_leg_bdc = Qlb.ModifiedFollowing
    fixed_leg_dc = Qlb.Actual360()
    spread = 0.0
    fixed_leg_schedule = Qlb.Schedule(start,
                                      end,
                                      fixed_leg_tenor,
                                      index.fixingCalendar(),
                                      fixed_leg_bdc,
                                      fixed_leg_bdc,
                                      Qlb.DateGeneration.Backward,
                                      False)
    floater_leg_schedule = Qlb.Schedule(start,
                                        end,
                                        index.tenor(),
                                        index.fixingCalendar(),
                                        index.businessDayConvention(),
                                        index.businessDayConvention(),
                                        Qlb.DateGeneration.Backward,
                                        False)
    swap = Qlb.VanillaSwap(typ,
                           notional,
                           fixed_leg_schedule,
                           fixed_rate,
                           fixed_leg_dc,
                           floater_leg_schedule,
                           index,
                           spread,
                           index.dayCounter())

    return swap, [index.fixingDate(x) for x in floater_leg_schedule][:-1]

def main():
    # setting evaluation date
    today = Qlb.Date(7, 4, 2015)
    Qlb.Settings.instance().setEvaluationDate(today)

    # setting market data
    rate = Qlb.SimpleQuote(0.03)
    rate_handle = Qlb.QuoteHandle(rate)
    daycount = Qlb.Actual365Fixed()
    yield_termstructure = Qlb.FlatForward(today, rate_handle, daycount)
    yield_termstructure.enableExtrapolation()
    yield_termstructure_handle = Qlb.RelinkableYieldTermStructureHandle(yield_termstructure)
    t0_yieldcurve = Qlb.YieldTermStructureHandle(yield_termstructure)
    euribor6M = Qlb.Euribor6M(yield_termstructure_handle)

    portfolio = [make_swap(today + Qlb.Period('2d'),
                           Qlb.Period('5y'),
                           1e8,
                           0.03,
                           euribor6M),
                 make_swap(today + Qlb.Period('2d'),
                           Qlb.Period('4y'),
                           5e6,
                           0.03,
                           euribor6M,
                           Qlb.VanillaSwap.Receiver)]
    engine = Qlb.DiscountingSwapEngine(yield_termstructure_handle)
    for deal, fixing_dates in portfolio:
        deal.setPricingEngine(engine)
        deal.NPV()
        print(deal.NPV())

    volatilities = [Qlb.QuoteHandle(Qlb.SimpleQuote(0.0075)),
         Qlb.QuoteHandle(Qlb.SimpleQuote(0.0075))]
    mean_reversion = [Qlb.QuoteHandle(Qlb.SimpleQuote(0.2))]
    model = Qlb.Gsr(t0_yieldcurve, [today+100], volatilities, mean_reversion, 16.)

    process = model.stateProcess()

    date_grid = [today + Qlb.Period(i, Qlb.Months) for i in range(0, 12*6)]
    for deal in portfolio:
        date_grid += deal[1]

    date_grid = np.unique(np.sort(date_grid))
    time_grid = np.vectorize(lambda x: Qlb.ActualActual().yearFraction(today, x))(date_grid)
    dt = time_grid[1:] - time_grid[:-1]

    # making random number generator
    seed = 1
    urng = Qlb.MersenneTwisterUniformRng(seed)
    usrg = Qlb.MersenneTwisterUniformRsg(len(time_grid)-1, urng)
    generator = Qlb.InvCumulativeMersenneTwisterGaussianRsg(usrg)

    N = 1500
    x = np.zeros((N, len(time_grid)))
    y = np.zeros((N, len(time_grid)))
    pillars = np.array([0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    zero_bonds = np.zeros((N, len(time_grid), 12))

    for j in range(12):
        zero_bonds[:, 0, j] = model.zerobond(pillars[j], 0, 0)
    for n in range(0, N):
        dWs = generator.nextSequence().value()
        for i in range(1, len(time_grid)):
            t0 = time_grid[i-1]
            t1 = time_grid[i]
            x[n, i] = process.expectation(t0,
                                         x[n, i-1],
                                         dt[i-1]) + dWs[i-1] * process.stdDeviation(t0,
                                                                                    x[n, i-1],
                                                                                    dt[i-1])
            y[n, i] = (x[n, i] - process.expectation(0, 0, t1)) / process.stdDeviation(0, 0, t1)
            for j in range(12):
                zero_bonds[n, i, j] = model.zerobond(t1+pillars[j],
                                                     t1,
                                                     y[n, i])
    for i in range(0, N):
        plt.plot(time_grid, x[i, :])

    #plt.show()


    npv_cube = np.zeros((N, len(date_grid), len(portfolio)))
    for p in range(0, N):
        for t in range(0, len(date_grid)):
            date = date_grid[t]
            Qlb.Settings.instance().setEvaluationDate(date)
            yieldcurve_dates = [date, date + Qlb.Period(6, Qlb.Months)]
            yieldcurve_dates += [date + Qlb.Period(i, Qlb.Years) for i in range(1, 11)]
            yieldcurve = Qlb.DiscountCurve(yieldcurve_dates,
                                           zero_bonds[p, t, :],
                                           Qlb.Actual365Fixed())
            yieldcurve.enableExtrapolation()
            yield_termstructure_handle.linkTo(yieldcurve)
            if euribor6M.isValidFixingDate(date):
                fixing = euribor6M.fixing(date)
                euribor6M.addFixing(date, fixing)
            for i in range(len(portfolio)):
                npv_cube[p, t, i] = portfolio[i][0].NPV()
        Qlb.IndexManager.instance().clearHistories()
    Qlb.Settings.instance().setEvaluationDate(today)
    yield_termstructure_handle.linkTo(yield_termstructure)

    discount_factors = np.vectorize(t0_yieldcurve.discount)(time_grid)
    discountednpv_cube = np.zeros(npv_cube.shape)
    for i in range(npv_cube.shape[2]):
        discountednpv_cube[:, :, i] = npv_cube[:, :, i] * discount_factors
    portfolio_npv = np.sum(npv_cube, axis=2)
    discounted_portfolio_npv = np.sum(discountednpv_cube, axis=2)

    # plot the first 30 NPV paths
    n_0 = 0
    n = 30
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    for i in range(n_0, n):
        ax1.plot(time_grid, portfolio_npv[i, :])
    for i in range(n_0, n):
        ax2.plot(time_grid, discounted_portfolio_npv[i, :])
    ax1.set_xlabel("Time in years")
    ax1.set_ylabel("NPV in time t Euros")
    ax1.set_title("Simulated npv paths")
    ax2.set_xlabel("Time in years")
    ax2.set_ylabel("NPV in time 0 Euros")
    ax2.set_title("Simulated discounted npv paths")
    #plt.show()

    print("portfolio_npv shape: ", portfolio_npv.shape)
    # Calculating the positive expected exposure
    E = portfolio_npv.copy()
    E[E<0] = 0
    EE = np.sum(E, axis=0) / N

    dE = discounted_portfolio_npv.copy()
    dE[dE<0] = 0
    dEE = np.sum(dE, axis=0) / N

    n = 30
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(time_grid, EE)
    ax2.plot(time_grid, dEE)
    ax1.set_xlabel("Time in years")
    ax1.set_ylabel("Exposure")
    ax1.set_title("Expected Exposure")
    ax2.set_xlabel("Time in years")
    ax2.set_ylabel("Discounted Exposure")
    ax2.set_title("Discounted Expected Exposure")

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(time_grid, dEE)
    plt.xlabel("Time in years")
    plt.ylabel("Discounting expected exposure")
    plt.ylim([-2000, 10000])
    plt.title("Expected exposure(netting set)")

    # Calculating PFE curve(95% tile)
    # pfe_curve = np.apply_along_axis(lambda x: np.sort(x)[0.95*N], 0, E)
    # Traceback (most recent call last):
    #   File "C:/Users/mmatt/01_work/25_xva/python/main.py", line 205, in <module>
    #     main()
    #   File "C:/Users/mmatt/01_work/25_xva/python/main.py", line 195, in main
    #     pfe_curve = np.apply_along_axis(lambda x: np.sort(x)[0.95*N], 0, E)
    #   File "C:\Users\mmatt\01_work\25_xva\python\venv\lib\site-packages\numpy\lib\shape_base.py", line 132, in apply_along_axis
    #     res = asanyarray(func1d(inarr_view[ind0], *args, **kwargs))
    #   File "C:/Users/mmatt/01_work/25_xva/python/main.py", line 195, in <lambda>
    #     pfe_curve = np.apply_along_axis(lambda x: np.sort(x)[0.95*N], 0, E)
    # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    # plt.figure(figsize=(7, 5), dpi=300)
    # plt.plot(time_grid, pfe_curve)
    # plt.xlabel("Time in years")
    # plt.ylabel("PFE")
    # plt.ylim([-2000, 35000])
    # plt.title("PFE (netting set)")
    # plt.show()

    # pfe = np.sort(np.max(E, axis=1))[0.95*N]
    # Traceback (most recent call last):
    #   File "C:/Users/mmatt/01_work/25_xva/python/main.py", line 217, in <module>
    #     if __name__ == '__main__':
    #   File "C:/Users/mmatt/01_work/25_xva/python/main.py", line 214, in main
    #     # pfe = np.sort(np.max(E, axis=1))[0.95*N]
    # IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
    # print(pfe)

    pd_dates = [today+Qlb.Period(i, Qlb.Years) for i in range(11)]
    hzrates = [0.02 * i for i in range(11)]
    pd_curve = Qlb.HazardRateCurve(pd_dates, hzrates, Qlb.Actual365Fixed())
    pd_curve.enableExtrapolation()

    # calculate default probabilities on grid times
    times = np.linspace(0, 30, 100)
    dp = np.vectorize(pd_curve.defaultProbability)(times)
    sp = np.vectorize(pd_curve.survivalProbability)(times)
    dd = np.vectorize(pd_curve.defaultDensity)(times)
    hr = np.vectorize(pd_curve.hazardRate)(times)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    ax1.plot(times, dp)
    ax2.plot(times, sp)
    ax3.plot(times, dd)
    ax4.plot(times, hr)
    ax1.set_xlabel("Time in years")
    ax2.set_xlabel("Time in years")
    ax3.set_xlabel("Time in years")
    ax4.set_xlabel("Time in years")
    ax1.set_ylabel("Probability")
    ax2.set_ylabel("Probability")
    ax3.set_ylabel("Density")
    ax4.set_ylabel("HazardRate")
    ax1.set_title("Default Probability")
    ax2.set_title("Survival Probability")
    ax3.set_title("Default Density")
    ax4.set_title("Hazard Rate")

    # calculate
    default_probabilities = np.vectorize(pd_curve.defaultProbability)
    dPD = default_probabilities(time_grid[:-1], time_grid[1:])

    # CVA
    recovery_rate = 0.4
    loss_given_default = 1 - recovery_rate
    cva = loss_given_default * np.sum(dEE[1:] * dPD)
    print("cva: ", cva)

if __name__ == '__main__':
    main()