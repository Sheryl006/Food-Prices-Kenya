# ==============================================================================
# CMT 429 Semester Project - Phase 2
# Food Prices in Kenya: Seasonal Patterns and Maize Price Forecasting
# Steps 3 & 4 of the Data Science Life Cycle: Data preparation + EDA
#
# Folder layout (create these next to this script):
#   data/raw/        <- put wfp_food_prices_ken.csv here, never edit it
#   data/processed/  <- cleaned files are saved here
#   figures/         <- plots are saved here
# ==============================================================================

# ---- 0. Setup ----------------------------------------------------------------
pkgs <- c("tidyverse", "lubridate", "janitor", "skimr", "sf", "forecast", "scales")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)

library(tidyverse)
library(lubridate)
library(janitor)
library(skimr)
library(sf)
library(forecast)
library(scales)

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

theme_set(theme_minimal(base_size = 12))
save_plot <- function(p, name, w = 9, h = 5.5) {
  ggsave(file.path("figures", name), p, width = w, height = h, dpi = 300)
}

# ==============================================================================
# STEP 2 (cont.): DATA UNDERSTANDING on the RAW file
# ==============================================================================

# ---- 1. Import ---------------------------------------------------------------
raw_path <- "data/raw/wfp_food_prices_ken.csv"
# If the file is not there yet, download it (needs internet). Record the date.
if (!file.exists(raw_path)) {
  download.file(
    "https://data.humdata.org/dataset/wfp-food-prices-for-kenya",  # replace with the direct CSV link from the HDX page
    raw_path, mode = "wb")
}
cat("File downloaded/used on:", format(Sys.Date()), "| size (MB):",
    round(file.size(raw_path) / 1e6, 2), "\n")

# Read everything as text first so the HXL tag row (row 2, e.g. #date) does not break types
raw <- read_csv(raw_path, col_types = cols(.default = col_character()))

# Drop the HXL row if present (its date cell starts with "#")
raw <- raw %>% filter(!str_starts(date, "#"))

raw <- raw %>%
  clean_names() %>%
  mutate(
    date      = ymd(date),
    price     = as.numeric(price),
    usdprice  = as.numeric(usdprice),
    latitude  = as.numeric(latitude),
    longitude = as.numeric(longitude)
  )

# ---- 2. Data understanding checks (report these in the document) ------------
dim(raw)
glimpse(raw)
summary(raw$price)
skim(raw)

cat("\nDate range:", format(min(raw$date)), "to", format(max(raw$date)), "\n")
raw %>% count(commodity, sort = TRUE) %>% print(n = 50)
raw %>% count(market, sort = TRUE) %>% print(n = 20)
raw %>% count(admin1, sort = TRUE)
raw %>% count(commodity, unit, pricetype) %>% arrange(commodity) %>% print(n = 100)
raw %>% count(priceflag)
raw %>% count(currency)

cat("\nMissing values per column:\n"); print(colSums(is.na(raw)))
cat("\nExact duplicate rows:", sum(duplicated(raw)), "\n")

# ==============================================================================
# STEP 3: DATA PREPARATION
# ==============================================================================

# ---- 3. Filter to maize and beans -------------------------------------------
# Exact commodity names verified against the downloaded file. We keep grain maize and
# the main dry beans; flour, green maize and the minor bean varieties (kidney, mung,
# yellow, dolichos) are excluded so that we compare like with like.
maize_names <- c("Maize", "Maize (white)", "Maize (white, dry)")
bean_names  <- c("Beans", "Beans (dry)", "Beans (rosecoco)")
is_target <- function(x) x %in% c(maize_names, bean_names)

mb <- raw %>%
  filter(is_target(commodity)) %>%
  distinct() %>%                                  # remove duplicates
  mutate(
    crop = if_else(str_detect(commodity, regex("maize", ignore_case = TRUE)),
                   "Maize", "Beans"),
    year = year(date),
    month = month(date, label = TRUE, abbr = TRUE)
  )

# ---- 4. Standardise units to KES per kg --------------------------------------
# unit examples: "KG", "90 KG", "50 KG". Extract the number of kg.
mb <- mb %>%
  mutate(
    kg_in_unit = case_when(
      str_detect(unit, regex("^kg$", ignore_case = TRUE)) ~ 1,
      str_detect(unit, regex("kg", ignore_case = TRUE))   ~ parse_number(unit),
      TRUE ~ NA_real_
    ),
    price_kg = price / kg_in_unit
  )

cat("\nRows dropped because unit cannot be converted to kg:",
    sum(is.na(mb$price_kg)), "\n")
mb <- mb %>% filter(!is.na(price_kg), price_kg > 0)

# Same commodity name can appear in different units across markets;
# after conversion everything is KES/kg and comparable.

# Several commodity names can describe the same crop in the same market and month
# (e.g. "Maize" and "Maize (white)"). Average them so each market has ONE price per month.
mb <- mb %>%
  mutate(ym = floor_date(date, "month")) %>%
  group_by(crop, pricetype, admin1, admin2, market, latitude, longitude, ym) %>%
  summarise(price_kg = mean(price_kg), .groups = "drop") %>%
  mutate(date = ym, year = year(date), month = month(date, label = TRUE, abbr = TRUE))

# ---- 5. Keep the study window (Jan 2020 - Dec 2024) -------------------------
mb_long <- mb                                                 # 2006-2024, for context chart only
mb_main <- mb %>% filter(date >= as.Date("2020-01-01"),
                         date <= as.Date("2024-12-31"))

cat("\nMain analysis rows:", nrow(mb_main), "\n")
mb_main %>% count(crop, pricetype)

# ---- 6. Missing months per market -------------------------------------------
coverage <- mb_main %>%
  group_by(crop, pricetype, market) %>%
  summarise(months_reported = n_distinct(floor_date(date, "month")),
            .groups = "drop") %>%
  arrange(months_reported)
print(coverage, n = 30)

# FINDING: reporting is very uneven. Only ~11 retail maize markets have 24+ months, and
# most have 1-5. The Phase 1 rule (>= 24 months) would keep almost nothing, so we
#  - use ALL observations for regional/market comparisons (RQ2), and
#  - use a "core" panel of markets with >= 12 months for trend/seasonality checks.
core_markets <- coverage %>% filter(months_reported >= 12)
mb_main <- mb_main %>%
  left_join(core_markets %>% select(crop, pricetype, market) %>% mutate(core = TRUE),
            by = c("crop", "pricetype", "market")) %>%
  mutate(core = coalesce(core, FALSE))

# How many markets report each month? (explains the jumps in the national average)
markets_per_month <- mb_main %>%
  group_by(crop, pricetype, date) %>% summarise(n_markets = n_distinct(market), .groups = "drop")
p_cov <- ggplot(markets_per_month, aes(date, n_markets, colour = pricetype)) +
  geom_line() + facet_wrap(~crop) +
  labs(title = "Number of markets reporting each month", x = NULL, y = "Markets", colour = NULL)
save_plot(p_cov, "00b_market_coverage.png")

# ---- 7. Outliers (IQR rule, per crop + price type) --------------------------
mb_main <- mb_main %>%
  group_by(crop, pricetype) %>%
  mutate(
    q1 = quantile(price_kg, 0.25), q3 = quantile(price_kg, 0.75),
    iqr = q3 - q1,
    outlier = price_kg < q1 - 1.5 * iqr | price_kg > q3 + 1.5 * iqr
  ) %>%
  ungroup()

mb_main %>% filter(outlier) %>% count(crop, pricetype, year)   # compare with 2022 drought

p_out <- ggplot(mb_main, aes(pricetype, price_kg, fill = crop)) +
  geom_boxplot(outlier.alpha = 0.4) +
  labs(title = "Price distribution and outliers", x = NULL, y = "KES per kg", fill = NULL)
save_plot(p_out, "00_outlier_boxplots.png")
# Decision: outliers that fall in 2022 are real drought spikes, so we KEEP them.
# Remove only values that are clearly errors (e.g., impossible prices) if you find any.

# ---- 8. Save cleaned data ----------------------------------------------------
write_csv(mb_main, "data/processed/maize_beans_2020_2024.csv")
write_csv(mb_long, "data/processed/maize_beans_2006_2024.csv")

# Main analysis focus: RETAIL prices (analyse wholesale the same way by changing the filter)
retail <- mb_main %>% filter(pricetype == "Retail")
wholesale <- mb_main %>% filter(pricetype == "Wholesale")
# NOTE: retail maize/bean prices have NO observations from Mar to Dec 2022 in this file.
# Wholesale prices do cover 2022, so the 2022 drought spike (RQ3) is shown with WHOLESALE.
retail %>% filter(year == 2022) %>% count(crop, month) %>% print()
wholesale %>% filter(year == 2022) %>% count(crop, month) %>% print(n = 30)

# ==============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ==============================================================================
# What the data checks showed (important for reading every chart below):
#  * Retail prices have a big gap: no maize observations Mar-Dec 2022.
#  * In Dec 2023 and Mar/Jun/Sep 2024 about 100+ markets report; in other months about 10.
#    Those large survey months have much higher average prices (about 84 vs 56 KES/kg for
#    retail maize), so the retail "national average" jumps with who is surveyed.
#  * Wholesale maize has data in every month (one gap), so it is the most reliable series.
# So: wholesale is used for seasonality, the 2022 drought and volatility; retail is used
# mainly for regional comparison (RQ2).

all_months <- seq(as.Date("2020-01-01"), as.Date("2024-12-01"), by = "month")

monthly_national <- function(df) {
  df %>%
    group_by(crop, date = floor_date(date, "month")) %>%
    summarise(price = mean(price_kg), n_markets = n_distinct(market), .groups = "drop") %>%
    complete(crop, date = all_months)      # missing months become NA so gaps show as gaps
}
nat_retail <- monthly_national(retail)
nat_whole  <- monthly_national(wholesale)
nat_both   <- bind_rows(mutate(nat_retail, pricetype = "Retail"),
                        mutate(nat_whole,  pricetype = "Wholesale")) %>%
  mutate(year = year(date), month = month(date, label = TRUE, abbr = TRUE))

# ---- RQ1: When do prices peak and drop? --------------------------------------
# Seasonal index = month price / that year's average * 100, averaged over years.
# Only years with >= 10 months of data are used (retail 2022 has 2 months, so it is excluded).
seasonal_index <- nat_both %>%
  filter(!is.na(price)) %>%
  group_by(pricetype, crop, year) %>% filter(n() >= 10) %>%
  mutate(index = price / mean(price) * 100) %>%
  group_by(pricetype, crop, month) %>%
  summarise(avg_price = mean(price), seasonal_index = mean(index),
            years_used = n_distinct(year), .groups = "drop")
print(seasonal_index, n = 48)

# Only report a calendar where at least 3 full years support it
price_calendar <- seasonal_index %>%
  filter(years_used >= 3) %>%
  group_by(pricetype, crop) %>%
  summarise(cheapest_month = month[which.min(seasonal_index)],
            cheapest_index = min(seasonal_index),
            peak_month = month[which.max(seasonal_index)],
            peak_index = max(seasonal_index),
            years_used = max(years_used), .groups = "drop")
print(price_calendar)
write_csv(price_calendar, "data/processed/price_calendar.csv")

p1a <- seasonal_index %>% filter(years_used >= 3) %>%
  ggplot(aes(month, seasonal_index, group = crop, colour = crop)) +
  geom_hline(yintercept = 100, linetype = "dashed") +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_wrap(~pricetype) +
  labs(title = "Seasonal price index by month (100 = annual average)",
       subtitle = "Kenya, 2020-2024; only years with 10+ months of data",
       x = NULL, y = "Index", colour = NULL)
save_plot(p1a, "01_seasonal_index.png", w = 11)

p1b <- nat_both %>% filter(!is.na(price)) %>%
  ggplot(aes(month, price, fill = crop)) +
  geom_boxplot(show.legend = FALSE) +
  facet_grid(crop ~ pricetype, scales = "free_y") +
  labs(title = "Price by calendar month", x = NULL, y = "KES per kg")
save_plot(p1b, "02_boxplot_by_month.png", w = 12, h = 7)

# STL decomposition: wholesale maize (complete months; the one gap is interpolated)
maize_w <- nat_whole %>% filter(crop == "Maize") %>% arrange(date)
maize_ts <- ts(forecast::na.interp(maize_w$price), start = c(2020, 1), frequency = 12)
png("figures/03_stl_maize.png", width = 2000, height = 1500, res = 250)
plot(stl(maize_ts, s.window = "periodic"),
     main = "STL decomposition: national wholesale maize price (KES/kg)")
dev.off()

# ---- RQ2: Regions and markets ------------------------------------------------
retail_geo <- retail %>% filter(!is.na(admin1))      # drops rows with no region
MIN_OBS <- 6   # a market needs at least 6 monthly prices to be ranked (1-2 prices is too thin)

region_stats <- retail_geo %>%
  group_by(crop, admin1) %>%
  summarise(mean_price = mean(price_kg), median_price = median(price_kg),
            n = n(), markets = n_distinct(market), .groups = "drop")
print(region_stats, n = 30)
write_csv(region_stats, "data/processed/region_stats.csv")

market_stats <- retail_geo %>%
  group_by(crop, admin1, admin2, market, latitude, longitude) %>%
  summarise(mean_price = mean(price_kg), median_price = median(price_kg),
            n = n(), .groups = "drop")
write_csv(market_stats, "data/processed/market_stats.csv")

ranked <- market_stats %>% filter(n >= MIN_OBS)
cat("\nMarkets with at least", MIN_OBS, "prices:\n"); print(count(ranked, crop))

top_bottom <- ranked %>%
  group_by(crop) %>%
  arrange(desc(mean_price), .by_group = TRUE) %>%
  filter(row_number() <= 10 | row_number() > n() - 10) %>%
  mutate(group = if_else(row_number() <= 10, "Most expensive", "Least expensive"),
         label = paste(market, crop, sep = "___")) %>%
  ungroup()
write_csv(top_bottom %>% select(-label), "data/processed/market_ranking.csv")

p2a <- ggplot(top_bottom, aes(reorder(label, mean_price), mean_price, fill = group)) +
  geom_col() + coord_flip() +
  scale_x_discrete(labels = function(x) str_remove(x, "___.*")) +
  facet_wrap(~crop, scales = "free") +
  labs(title = paste0("10 most and least expensive markets (markets with ", MIN_OBS, "+ prices)"),
       x = NULL, y = "Average KES per kg", fill = NULL)
save_plot(p2a, "04_market_ranking.png", w = 12, h = 6)

gap <- ranked %>%
  group_by(crop) %>%
  summarise(max_market = market[which.max(mean_price)], max_price = max(mean_price),
            min_market = market[which.min(mean_price)], min_price = min(mean_price),
            gap_kes = max_price - min_price, ratio = max_price / min_price)
print(gap)

p2b <- retail_geo %>%
  group_by(crop, admin1, date = floor_date(date, "month")) %>%
  summarise(price = mean(price_kg), .groups = "drop") %>%
  ggplot(aes(date, price, colour = crop)) +
  geom_line() + geom_point(size = 0.8) + facet_wrap(~admin1) +
  labs(title = "Monthly retail price by region", x = NULL, y = "KES per kg", colour = NULL)
save_plot(p2b, "05_region_facets.png", w = 12, h = 8)

markets_sf <- market_stats %>%
  filter(!is.na(latitude), !is.na(longitude)) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

p2c <- ggplot()
if (requireNamespace("rnaturalearth", quietly = TRUE)) {
  kenya <- rnaturalearth::ne_countries(country = "Kenya", scale = "medium", returnclass = "sf")
  p2c <- p2c + geom_sf(data = kenya, fill = "grey95", colour = "grey50")
}
p2c <- p2c +
  geom_sf(data = markets_sf, aes(colour = mean_price), size = 3) +
  scale_colour_viridis_c(option = "plasma", direction = -1) +
  facet_wrap(~crop) +
  labs(title = "Average retail price by market, 2020-2024", colour = "KES/kg")
save_plot(p2c, "06_market_map.png", w = 11, h = 6)

# ---- RQ3: Trend 2020-2024 and long-term context ------------------------------
drought_band <- annotate("rect", xmin = as.Date("2022-01-01"), xmax = as.Date("2022-12-31"),
                         ymin = -Inf, ymax = Inf, alpha = 0.1, fill = "red")

p3a <- ggplot(nat_retail, aes(date, price, colour = crop)) +
  geom_line(linewidth = 1) + drought_band +
  labs(title = "National average RETAIL price, Jan 2020 - Dec 2024",
       subtitle = "Gaps are months with no data; spikes in late 2023-2024 are large survey rounds",
       x = NULL, y = "KES per kg", colour = NULL)
save_plot(p3a, "07_trend_2020_2024.png")

p3w <- ggplot(nat_whole, aes(date, price, colour = crop)) +
  geom_line(linewidth = 1) + drought_band +
  labs(title = "National average WHOLESALE price, Jan 2020 - Dec 2024",
       subtitle = "Most reliable series; beans have few months of data (gaps shown)",
       x = NULL, y = "KES per kg", colour = NULL)
save_plot(p3w, "07b_trend_wholesale.png")

# Yearly summary: only years with >= 10 months are treated as reliable
yearly <- nat_both %>%
  group_by(pricetype, crop, year) %>%
  summarise(months_with_data = sum(!is.na(price)),
            avg_price = mean(price, na.rm = TRUE), sd = sd(price, na.rm = TRUE),
            cv_pct = sd / avg_price * 100, .groups = "drop") %>%
  mutate(reliable = months_with_data >= 10) %>%
  group_by(pricetype, crop) %>% arrange(year, .by_group = TRUE) %>%
  mutate(yoy_pct = if_else(reliable & lag(reliable), (avg_price / lag(avg_price) - 1) * 100, NA_real_)) %>%
  ungroup()
print(yearly, n = 30)
write_csv(yearly, "data/processed/yearly_summary.csv")

p3b <- yearly %>% filter(reliable) %>%
  ggplot(aes(factor(year), cv_pct, fill = crop)) +
  geom_col(position = "dodge") + facet_wrap(~pricetype) +
  labs(title = "Price volatility by year (coefficient of variation of monthly prices)",
       subtitle = "Years with fewer than 10 months of data are not shown",
       x = NULL, y = "CV (%)", fill = NULL)
save_plot(p3b, "08_volatility.png", w = 11)

# Long-term context 2006-2024 (context only)
long_maize <- mb_long %>%
  filter(crop == "Maize", date <= as.Date("2024-12-31")) %>%
  group_by(pricetype, date = floor_date(date, "month")) %>%
  summarise(price = mean(price_kg), .groups = "drop") %>%
  complete(pricetype, date = seq(as.Date("2006-01-01"), as.Date("2024-12-01"), by = "month"))

p3c <- ggplot(long_maize, aes(date, price)) +
  geom_line(colour = "#b5651d") + facet_wrap(~pricetype, ncol = 1) +
  geom_vline(xintercept = as.Date(c("2011-07-01", "2017-03-01", "2022-08-01")),
             linetype = "dashed", colour = "grey40") +
  labs(title = "National average maize price, 2006-2024",
       subtitle = "Dashed lines: approximate drought peaks (2011, 2017, 2022). Nominal prices.",
       x = NULL, y = "KES per kg")
save_plot(p3c, "09_longterm_maize.png")

# How big was each drought spike? (wholesale, nominal)
peak_in <- function(from, to, label) {
  long_maize %>%
    filter(pricetype == "Wholesale", !is.na(price),
           date >= as.Date(from), date <= as.Date(to)) %>%
    slice_max(price, n = 1) %>% mutate(drought = label)
}
drought_compare <- bind_rows(peak_in("2010-06-01", "2012-06-01", "2011"),
                             peak_in("2016-06-01", "2018-06-01", "2017"),
                             peak_in("2021-12-01", "2024-06-01", "2022-23"))
print(drought_compare)
write_csv(drought_compare, "data/processed/drought_compare.csv")

cat("\nPhase 2 script finished. Check the figures/ and data/processed/ folders.\n")
