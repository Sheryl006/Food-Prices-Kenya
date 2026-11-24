# ==============================================================================
# CMT 429 Semester Project - Phase 3
# Food Prices in Kenya: Seasonal Patterns and Maize Price Forecasting
# Steps 5 & 6 of the Data Science Life Cycle: Modelling + Evaluation (RQ4)
#
# Run this from the Phase2 project folder (the one that holds data/ and figures/),
# after phase2_analysis.R. It reads the raw WFP file again so that it can also use
# the months after December 2024 as an extra check and for a real forecast.
# ==============================================================================

# ---- 0. Setup ----------------------------------------------------------------
pkgs <- c("tidyverse", "lubridate", "forecast", "randomForest", "scales")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)

library(tidyverse)
library(lubridate)
library(forecast)
library(randomForest)
library(scales)

set.seed(429)                                   # random forest results repeat exactly
dir.create("figures", showWarnings = FALSE)
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
save_plot <- function(p, name, w = 10, h = 6) {
  ggsave(file.path("figures", name), p, width = w, height = h, dpi = 300, bg = "white")
}
theme_set(theme_minimal(base_size = 12))

# ---- 1. Rebuild the national wholesale maize series ----------------------------
# Same cleaning rules as Phase 2: grain maize only, prices converted to KES per kg,
# one price per market per month, then the national average across markets.
raw <- read_csv("data/raw/wfp_food_prices_ken.csv",
                col_types = cols(.default = col_character())) %>%
  filter(!str_starts(date, "#")) %>%
  mutate(date = ymd(date), price = as.numeric(price))

maize_names <- c("Maize", "Maize (white)", "Maize (white, dry)")

maize_w_markets <- raw %>%
  filter(commodity %in% maize_names, pricetype == "Wholesale") %>%
  distinct() %>%
  mutate(kg_in_unit = suppressWarnings(case_when(          # "KG" alone gives a harmless
           str_detect(unit, regex("^kg$", ignore_case = TRUE)) ~ 1,   # parse warning
           str_detect(unit, regex("kg", ignore_case = TRUE))   ~ parse_number(unit),
           TRUE ~ NA_real_)),
         price_kg = price / kg_in_unit) %>%
  filter(!is.na(price_kg), price_kg > 0) %>%
  mutate(date = floor_date(date, "month")) %>%
  group_by(market, date) %>%
  summarise(price_kg = mean(price_kg), .groups = "drop")

last_month <- max(maize_w_markets$date)
all_months <- seq(as.Date("2020-01-01"), last_month, by = "month")

national <- maize_w_markets %>%
  filter(date >= as.Date("2020-01-01")) %>%
  group_by(date) %>%
  summarise(price = mean(price_kg), n_markets = n_distinct(market), .groups = "drop") %>%
  complete(date = all_months)

cat("Series runs from", format(min(national$date)), "to", format(last_month), "\n")
cat("Months with no wholesale maize price:\n")
print(national %>% filter(is.na(price)) %>% pull(date))

# Fill single missing months by interpolation (needed to build lags), and keep a flag
national <- national %>%
  mutate(filled = is.na(price),
         price = as.numeric(na.interp(ts(price, frequency = 12))))

# ---- 2. Features: last 1, 2 and 3 months' prices + calendar month -------------
model_data <- national %>%
  arrange(date) %>%
  mutate(lag1 = lag(price, 1), lag2 = lag(price, 2), lag3 = lag(price, 3),
         month = factor(month(date), levels = 1:12),
         month_num = month(date),
         year = year(date)) %>%
  filter(!is.na(lag3))

train <- model_data %>% filter(date <= as.Date("2023-12-01"))              # 2020-2023
test  <- model_data %>% filter(date >= as.Date("2024-01-01"),
                               date <= as.Date("2024-12-01"))              # 2024
cat("\nTraining months:", nrow(train), "| Test months:", nrow(test), "\n")

# ---- 3. Models -------------------------------------------------------------------
# (a) Naive baseline: next month's price = this month's price
naive_pred <- function(df) df$lag1

# (b) Multiple linear regression on lags and month
fit_lm <- lm(price ~ lag1 + lag2 + lag3 + month, data = train)
summary(fit_lm)

# (c) ARIMA chosen automatically on the training months
train_ts <- ts(national$price[national$date <= as.Date("2023-12-01")],
               start = c(2020, 1), frequency = 12)
fit_arima <- auto.arima(train_ts)
print(fit_arima)

# One-step-ahead ARIMA forecasts: keep the fitted model fixed and feed it each new month
one_step_arima <- function(fit, series_to, start_test) {
  full_ts <- ts(national$price[national$date <= series_to], start = c(2020, 1), frequency = 12)
  refit <- Arima(full_ts, model = fit)                   # same coefficients, no re-estimation
  fc <- fitted(refit)                                    # one-step forecasts for every month
  as.numeric(fc)[national$date[national$date <= series_to] >= start_test]
}

# (d) Random forest on lags and month
fit_rf <- randomForest(price ~ lag1 + lag2 + lag3 + month_num, data = train,
                       ntree = 500, importance = TRUE)
print(fit_rf)

# ---- 4. Evaluate on 2024 ----------------------------------------------------------
metrics <- function(actual, pred) {
  tibble(RMSE = sqrt(mean((actual - pred)^2)),
         MAE  = mean(abs(actual - pred)),
         MAPE = mean(abs(actual - pred) / actual) * 100)
}

pred_2024 <- test %>%
  select(date, actual = price, filled) %>%
  mutate(Naive      = naive_pred(test),
         Regression = predict(fit_lm, newdata = test),
         ARIMA      = one_step_arima(fit_arima, as.Date("2024-12-01"), as.Date("2024-01-01")),
         `Random forest` = predict(fit_rf, newdata = test))

results_2024 <- pred_2024 %>%
  pivot_longer(Naive:`Random forest`, names_to = "model", values_to = "pred") %>%
  group_by(model) %>%
  summarise(metrics(actual, pred), .groups = "drop") %>%
  arrange(RMSE)
cat("\nTest results, January to December 2024 (KES per kg):\n")
print(results_2024)
write_csv(results_2024, "data/processed/model_results_2024.csv")
write_csv(pred_2024, "data/processed/predictions_2024.csv")

# ---- 5. Extra check: retrain on 2020-2024, test on every month after 2024 ---------
train2 <- model_data %>% filter(date <= as.Date("2024-12-01"))
test2  <- model_data %>% filter(date >= as.Date("2025-01-01"))
fit_lm2 <- lm(price ~ lag1 + lag2 + lag3 + month, data = train2)
fit_arima2 <- auto.arima(ts(national$price[national$date <= as.Date("2024-12-01")],
                            start = c(2020, 1), frequency = 12))
fit_rf2 <- randomForest(price ~ lag1 + lag2 + lag3 + month_num, data = train2, ntree = 500)

pred_after <- test2 %>%
  select(date, actual = price, filled) %>%
  mutate(Naive      = naive_pred(test2),
         Regression = predict(fit_lm2, newdata = test2),
         ARIMA      = one_step_arima(fit_arima2, last_month, as.Date("2025-01-01")),
         `Random forest` = predict(fit_rf2, newdata = test2))

results_after <- pred_after %>%
  filter(!filled) %>%                                   # score only real observed months
  pivot_longer(Naive:`Random forest`, names_to = "model", values_to = "pred") %>%
  group_by(model) %>%
  summarise(metrics(actual, pred), .groups = "drop") %>%
  arrange(RMSE)
cat("\nExtra check, January 2025 to", format(last_month, "%B %Y"), ":\n")
print(results_after)
write_csv(results_after, "data/processed/model_results_2025_on.csv")
write_csv(pred_after, "data/processed/predictions_2025_on.csv")

# ---- 6. Figures ---------------------------------------------------------------------
model_cols <- c(Actual = "black", Naive = "#999999", Regression = "#1b9e77",
                ARIMA = "#d95f02", `Random forest` = "#7570b3")

p_test <- pred_2024 %>%
  pivot_longer(c(actual, Naive:`Random forest`), names_to = "series", values_to = "price") %>%
  mutate(series = recode(series, actual = "Actual")) %>%
  ggplot(aes(date, price, colour = series, linewidth = series == "Actual")) +
  geom_line() + geom_point(size = 1.5) +
  scale_linewidth_manual(values = c(`TRUE` = 1.4, `FALSE` = 0.7), guide = "none") +
  scale_colour_manual(values = model_cols) +
  scale_x_date(date_labels = "%b", date_breaks = "1 month") +
  labs(title = "Actual vs predicted wholesale maize price, test year 2024",
       subtitle = "One-month-ahead forecasts, trained on 2020 to 2023 (ARIMA chose a random walk, so it equals Naive)",
       x = NULL, y = "KES per kg", colour = NULL)
save_plot(p_test, "10_actual_vs_predicted_2024.png")

p_err <- bind_rows(mutate(results_2024, period = "Test 2024"),
                   mutate(results_after, period = "Check 2025 onwards")) %>%
  mutate(period = factor(period, levels = c("Test 2024", "Check 2025 onwards")),
         model = fct_reorder(model, RMSE)) %>%
  ggplot(aes(model, MAPE, fill = model)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f%%", MAPE)), vjust = -0.4, size = 3.5) +
  facet_wrap(~period) +
  scale_fill_manual(values = model_cols) +
  labs(title = "Average forecast error by model (MAPE, lower is better)",
       x = NULL, y = "MAPE (%)")
save_plot(p_err, "11_model_errors.png", w = 11)

p_after <- pred_after %>%
  select(date, actual, Naive, Regression, `Random forest`) %>%
  pivot_longer(-date, names_to = "series", values_to = "price") %>%
  mutate(series = recode(series, actual = "Actual")) %>%
  ggplot(aes(date, price, colour = series)) +
  geom_line(linewidth = 0.9) + geom_point(size = 1.3) +
  scale_colour_manual(values = model_cols) +
  labs(title = "Extra check: forecasts for months after 2024",
       subtitle = "Models retrained on 2020 to 2024 and never shown these months",
       x = NULL, y = "KES per kg", colour = NULL)
save_plot(p_after, "12_check_2025_on.png")

# ---- 7. Final model and a real forecast for the next 3 months ---------------------
# The tests show that last month's price is the best single predictor, and auto.arima
# also chose a random walk, ARIMA(0,1,0). So the final model is ARIMA refitted on ALL
# months; its point forecast equals the last observed price, and its intervals show
# how uncertain the next months are.
full_ts <- ts(national$price, start = c(2020, 1), frequency = 12)
fit_final <- auto.arima(full_ts)
print(fit_final)
fc <- forecast(fit_final, h = 3, level = c(80, 95))
fc_months <- last_month %m+% months(1:3)
forecast_next <- tibble(month = format(fc_months, "%B %Y"),
                        forecast = round(as.numeric(fc$mean), 1),
                        low_80 = round(as.numeric(fc$lower[, "80%"]), 1),
                        high_80 = round(as.numeric(fc$upper[, "80%"]), 1),
                        low_95 = round(as.numeric(fc$lower[, "95%"]), 1),
                        high_95 = round(as.numeric(fc$upper[, "95%"]), 1))
cat("\nLast observed month:", format(last_month, "%B %Y"), "=",
    round(tail(national$price, 1), 1), "KES per kg\n")
cat("Forecast for the next 3 months (KES per kg):\n")
print(forecast_next)
write_csv(forecast_next, "data/processed/forecast_next_months.csv")

fc_df <- tibble(date = fc_months, mean = as.numeric(fc$mean),
                lo80 = as.numeric(fc$lower[, "80%"]), hi80 = as.numeric(fc$upper[, "80%"]),
                lo95 = as.numeric(fc$lower[, "95%"]), hi95 = as.numeric(fc$upper[, "95%"]))
p_fc <- national %>%
  filter(date >= as.Date("2023-01-01")) %>%
  ggplot(aes(date, price)) +
  geom_ribbon(data = fc_df, aes(x = date, ymin = lo95, ymax = hi95), inherit.aes = FALSE,
              fill = "#d95f02", alpha = 0.15) +
  geom_ribbon(data = fc_df, aes(x = date, ymin = lo80, ymax = hi80), inherit.aes = FALSE,
              fill = "#d95f02", alpha = 0.3) +
  geom_line(linewidth = 0.9) +
  geom_line(data = fc_df, aes(date, mean), colour = "#d95f02", linewidth = 0.9) +
  geom_point(data = fc_df, aes(date, mean), colour = "#d95f02", size = 2) +
  labs(title = "Forecast of national wholesale maize price for the next 3 months",
       subtitle = "Random-walk ARIMA(0,1,0); dark band 80%, light band 95% prediction interval",
       x = NULL, y = "KES per kg")
save_plot(p_fc, "13_next_months_forecast.png")

# ---- 8. Insight: what happened after the study window? -----------------------------
yearly_after <- national %>%
  mutate(year = year(date)) %>%
  group_by(year) %>%
  summarise(months = sum(!filled), avg_price = mean(price[!filled]), .groups = "drop")
print(yearly_after)
write_csv(yearly_after, "data/processed/yearly_wholesale_maize_2020_on.csv")

cat("\nPhase 3 script finished. See figures/10 to 13 and data/processed/.\n")
