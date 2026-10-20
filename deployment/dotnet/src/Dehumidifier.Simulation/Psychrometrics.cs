namespace Dehumidifier.Simulation;

/// <summary>
/// 与 Python 基准逐项对应的水面湿空气公式（docs/requirement.txt 企业参考代码契约）。
/// 饱和压：&lt;=0 ℃ 用企业冰面基准多项式，&gt;0 ℃ 用液态水多项式（两者与参考代码
/// CalculateAH/CalculateRH 逐系数一致）；水汽比 0.62198。
/// 露点 ↔ 含湿量：露点 &lt;=0 ℃（W &lt;=3.764）用 DptAH 表分段线性插值；
/// 超出表范围按同一饱和压公式数值反演；W&lt;0.0001 按最后一段线性外推；W=0 为 -inf 干燥极限。
/// </summary>
public static class Psychrometrics
{
    public const double P_ATM = 101325.0;
    public const double SaturationWaterRatio = 0.62198;
    public const double ConfigWaterRatio = 0.622;

    /// <summary>企业 DptAH 映射表：DptAhTable[i] = 露点 -i ℃ 对应含湿量 (g/kg)，88 项覆盖 0 ~ -87 ℃。</summary>
    internal static readonly double[] DptAhTable =
    {
        3.764, 3.466, 3.189, 2.933, 2.694, 2.474, 2.271, 2.083, 1.91, 1.749,
        1.601, 1.464, 1.338, 1.223, 1.116, 1.018, 0.928, 0.845, 0.769, 0.741,
        0.636, 0.578, 0.522, 0.475, 0.431, 0.39, 0.353, 0.319, 0.288, 0.26,
        0.233, 0.211, 0.19, 0.171, 0.153, 0.138, 0.123, 0.111, 0.099, 0.089,
        0.079, 0.071, 0.063, 0.056, 0.0499, 0.0444, 0.0394, 0.035, 0.031, 0.0274,
        0.0243, 0.0214, 0.0189, 0.0167, 0.0147, 0.0129, 0.0113, 0.0099, 0.0087, 0.0076,
        0.0067, 0.0058, 0.0051, 0.0044, 0.0038, 0.0033, 0.0029, 0.0025, 0.0022, 0.0019,
        0.0016, 0.0014, 0.0012, 0.001, 0.0009, 0.00075, 0.00064, 0.00055, 0.00047, 0.0004,
        0.00034, 0.00029, 0.00024, 0.0002, 0.00017, 0.00015, 0.00012, 0.0001,
    };

    private const double DptAhMax = 3.764;    // 表上界：露点 0 ℃
    private const double DptAhMin = 0.0001;   // 表下界：露点 -87 ℃
    private static readonly double DptAhLowestC = -(DptAhTable.Length - 1.0);
    /// <summary>低于表下限时按最后一段（-87 ~ -86 ℃）线性外推的斜率 dTdp/dW。</summary>
    private static readonly double DptAhExtrapolationSlope =
        (DptAhLowestC + 1.0 - DptAhLowestC) / (DptAhTable[^2] - DptAhTable[^1]);

    public static double SaturationVaporPressureWaterPa(double temperatureC)
    {
        if (!double.IsFinite(temperatureC) || temperatureC + 273.15 <= 0) throw new SimulationException("INVALID_PHYSICS", "饱和压计算温度必须高于绝对零度且有限");
        var absolute = temperatureC + 273.15;
        double lnPressure;
        if (temperatureC <= 0)
        {
            // 2026-09-17：企业参考代码 DBT <= 0 分支（冰面基准多项式）。
            lnPressure = -5674.5359 / absolute + 6.3925247 - 9.677843e-3 * absolute + 6.2215701e-7 * absolute * absolute
                + 2.0747825e-9 * absolute * absolute * absolute - 9.484024e-13 * Math.Pow(absolute, 4) + 4.1635 * Math.Log(absolute);
        }
        else
        {
            lnPressure = -5.8002206e3 / absolute + 1.3914993 - 4.8640239e-2 * absolute + 4.1764768e-5 * absolute * absolute
                - 1.4452093e-8 * absolute * absolute * absolute + 6.5459673 * Math.Log(absolute);
        }
        return Math.Exp(lnPressure);
    }

    public static double SaturationHumidityRatioGPerKg(double temperatureC, double pressurePa = P_ATM)
    {
        ValidatePressure(pressurePa);
        var pws = SaturationVaporPressureWaterPa(temperatureC);
        return pws < pressurePa ? 1000.0 * SaturationWaterRatio * pws / (pressurePa - pws) : double.PositiveInfinity;
    }

    public static double DewPointWaterC(double humidityGPerKg, double pressurePa = P_ATM)
    {
        ValidatePressure(pressurePa);
        if (!double.IsFinite(humidityGPerKg) || humidityGPerKg < 0) throw new SimulationException("INVALID_PHYSICS", "露点输入含湿量必须非负且有限");
        if (humidityGPerKg == 0) return double.NegativeInfinity;
        if (humidityGPerKg <= DptAhMax)
        {
            if (humidityGPerKg >= DptAhMin)
            {
                // 表内：二分找到 Table[tableLow] >= W >= Table[tableHigh]（tableHigh = tableLow+1），线性插值。
                var tableLow = 0;
                var tableHigh = DptAhTable.Length - 1;
                while (tableHigh - tableLow > 1)
                {
                    var mid = (tableLow + tableHigh) / 2;
                    if (DptAhTable[mid] >= humidityGPerKg) tableLow = mid; else tableHigh = mid;
                }
                var wHigh = DptAhTable[tableLow];
                var wLow = DptAhTable[tableHigh];
                return -tableLow - (wHigh - humidityGPerKg) / (wHigh - wLow);
            }
            // 低于表下限：按最后一段斜率线性外推（与 Python 基准一致）。
            return DptAhLowestC + (humidityGPerKg - DptAhMin) * DptAhExtrapolationSlope;
        }
        // 超出映射表：同一饱和压公式数值反演。区间下界延伸到 0 ℃ 以下，
        // 覆盖表上界 3.764 与公式 0 ℃ 饱和含湿量（约 3.774）之间的窄缝及低压工况。
        var vaporPressure = pressurePa * humidityGPerKg / (1000.0 * SaturationWaterRatio + humidityGPerKg);
        var lowPressure = SaturationVaporPressureWaterPa(-150.0);
        var highPressure = SaturationVaporPressureWaterPa(100.0);
        if (vaporPressure < lowPressure || vaporPressure > highPressure) throw new SimulationException("INVALID_PHYSICS", $"水面露点超出数值反演范围: vapor_pressure={vaporPressure:G17} Pa");
        var low = -150.0; var high = 100.0;
        for (var i = 0; i < 100; i++)
        {
            var mid = (low + high) / 2.0;
            if (SaturationVaporPressureWaterPa(mid) < vaporPressure) low = mid; else high = mid;
        }
        return (low + high) / 2.0;
    }

    public static double HumidityRatioFromDewPointC(double dewPointC, double pressurePa = P_ATM)
    {
        ValidatePressure(pressurePa);
        if (double.IsNegativeInfinity(dewPointC)) return 0.0;
        if (!double.IsFinite(dewPointC)) throw new SimulationException("INVALID_PHYSICS", "露点输入必须有限（-inf 干燥极限除外）");
        if (dewPointC <= 0)
        {
            if (dewPointC >= DptAhLowestC)
            {
                var index = (int)Math.Floor(-dewPointC);
                if (index > DptAhTable.Length - 2) index = DptAhTable.Length - 2;
                var fraction = -dewPointC - index;
                return (1.0 - fraction) * DptAhTable[index] + fraction * DptAhTable[index + 1];
            }
            // 低于表下限：按最后一段斜率线性外推。
            var humidity = DptAhMin + (dewPointC - DptAhLowestC) / DptAhExtrapolationSlope;
            if (humidity <= 0) throw new SimulationException("INVALID_PHYSICS", "露点超出线性外推有效范围，含湿量将为非正值");
            return humidity;
        }
        return SaturationHumidityRatioGPerKg(dewPointC, pressurePa);
    }

    public static double EnthalpyKJ(double temperatureC, double humidityGPerKg)
        => 1.006 * temperatureC + (humidityGPerKg / 1000.0) * (2501.0 + 1.86 * temperatureC);

    public static double TemperatureFromEnthalpy(double enthalpyKJ, double humidityGPerKg)
    {
        var d = humidityGPerKg / 1000.0;
        return (enthalpyKJ - 2501.0 * d) / (1.006 + 1.86 * d);
    }

    public static double RelativeHumidity(double temperatureC, double humidityGPerKg, double pressurePa = P_ATM)
    {
        ValidatePressure(pressurePa);
        var humidity = Math.Max(humidityGPerKg, 0.0);
        var vaporPressure = pressurePa * humidity / (1000.0 * SaturationWaterRatio + humidity);
        var result = vaporPressure / SaturationVaporPressureWaterPa(temperatureC);
        return Math.Clamp(result, 0.0, 1.0);
    }

    private static void ValidatePressure(double pressurePa)
    {
        if (!double.IsFinite(pressurePa) || pressurePa <= 0) throw new SimulationException("INVALID_PHYSICS", "系统总压必须是正的有限值");
    }
}
