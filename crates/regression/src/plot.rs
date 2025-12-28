use argmin::core::Error;
use plotters::prelude::*;

const SCALE: f32 = 2.0;
const BASE_CANVAS: (u32, u32) = (2560, 1440);
const BASE_MARGIN: i32 = 30;
const BASE_LABEL_AREA: i32 = 70;
const BASE_POINT_RADIUS: i32 = 6;
const BASE_LINE_WIDTH: u32 = 6;
const BASE_LEGEND_POINT_RADIUS: i32 = 4;
const BASE_LABEL_FONT: i32 = 28;
const BASE_AXIS_DESC_FONT: i32 = 34;
const BASE_SERIES_LABEL_FONT: i32 = 48;
const BASE_LEGEND_LINE_HALF: i32 = 10;
pub const POINTS_SVG: &str = "points.svg";
pub const FIT_SVG: &str = "fit.svg";

fn scaled_i32(v: i32) -> i32 {
    (v as f32 * SCALE) as i32
}

fn scaled_u32(v: u32) -> u32 {
    (v as f32 * SCALE) as u32
}

fn canvas_size() -> (u32, u32) {
    (
        (BASE_CANVAS.0 as f32 * SCALE) as u32,
        (BASE_CANVAS.1 as f32 * SCALE) as u32,
    )
}

fn bounds_with_model<F>(temps: &[f32], model: F) -> (f32, f32, f32)
where
    F: Fn(f32) -> f32,
{
    let x_max = temps.len() as f32;

    // Захватываем диапазон по y: реальные точки и предсказанная кривая.
    let mut ys: Vec<f32> = temps.to_vec();
    ys.extend((0..temps.len()).map(|i| model(i as f32)));
    let (min_y, max_y) = ys
        .iter()
        .fold((f32::MAX, f32::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
    let pad = ((max_y - min_y) * 0.1).max(1.0);
    (x_max, min_y - pad, max_y + pad)
}

fn draw_points_only(
    temps: &[f32],
    x_max: f32,
    min_y: f32,
    max_y: f32,
    path: &str,
) -> Result<(), Error> {
    let root = SVGBackend::new(path, canvas_size()).into_drawing_area();
    root.fill(&TRANSPARENT)
        .map_err(|e| Error::msg(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(scaled_i32(BASE_MARGIN))
        .set_label_area_size(LabelAreaPosition::Left, scaled_i32(BASE_LABEL_AREA))
        .set_label_area_size(
            LabelAreaPosition::Bottom,
            scaled_i32(BASE_LABEL_AREA),
        )
        .build_cartesian_2d(0f32..x_max, min_y..max_y)
        .map_err(|e| Error::msg(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("День")
        .y_desc("Температура")
        .label_style(("sans-serif", scaled_i32(BASE_LABEL_FONT)))
        .axis_desc_style(("sans-serif", scaled_i32(BASE_AXIS_DESC_FONT)))
        .draw()
        .map_err(|e| Error::msg(e.to_string()))?;

    // Точки данных.
    chart
        .draw_series(
            temps
                .iter()
                .enumerate()
                .map(|(i, &temp)| {
                    Circle::new((i as f32, temp), scaled_i32(BASE_POINT_RADIUS), BLUE.filled())
                }),
        )
        .map_err(|e| Error::msg(e.to_string()))?
        .label("данные")
        .legend(|(x, y)| {
            Circle::new((x, y), scaled_i32(BASE_LEGEND_POINT_RADIUS), BLUE.filled())
        });

    root.present().map_err(|e| Error::msg(e.to_string()))?;

    Ok(())
}

pub fn save_plot_with_model<F>(temps: &[f32], model: F) -> Result<(), Error>
where
    F: Fn(f32) -> f32,
{
    let (x_max, min_y, max_y) = bounds_with_model(temps, &model);

    // 1) Только точки.
    draw_points_only(temps, x_max, min_y, max_y, POINTS_SVG)?;

    // 2) Точки + модель.
    let root = SVGBackend::new(FIT_SVG, canvas_size()).into_drawing_area();
    root.fill(&TRANSPARENT)
        .map_err(|e| Error::msg(e.to_string()))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(scaled_i32(BASE_MARGIN))
        .set_label_area_size(LabelAreaPosition::Left, scaled_i32(BASE_LABEL_AREA))
        .set_label_area_size(
            LabelAreaPosition::Bottom,
            scaled_i32(BASE_LABEL_AREA),
        )
        .build_cartesian_2d(0f32..x_max, min_y..max_y)
        .map_err(|e| Error::msg(e.to_string()))?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("День")
        .y_desc("Температура")
        .label_style(("sans-serif", scaled_i32(BASE_LABEL_FONT)))
        .axis_desc_style(("sans-serif", scaled_i32(BASE_AXIS_DESC_FONT)))
        .draw()
        .map_err(|e| Error::msg(e.to_string()))?;

    let model_line_style = ShapeStyle {
        color: RED.mix(1.0),
        filled: false,
        stroke_width: scaled_u32(BASE_LINE_WIDTH),
    };
    let legend_line_style = model_line_style.clone();

    // Кривая модели толще для читаемости на слайдах.
    chart
        .draw_series(LineSeries::new(
            (0..temps.len()).map(|i| (i as f32, model(i as f32))),
            model_line_style.clone(),
        ))
        .map_err(|e| Error::msg(e.to_string()))?
        .label("модель")
        .legend(move |(x, y)| {
            PathElement::new(
                vec![
                    (x - scaled_i32(BASE_LEGEND_LINE_HALF), y),
                    (x + scaled_i32(BASE_LEGEND_LINE_HALF), y),
                ],
                legend_line_style.clone(),
            )
        });

    // Точки данных.
    chart
        .draw_series(
            temps
                .iter()
                .enumerate()
                .map(|(i, &temp)| {
                    Circle::new((i as f32, temp), scaled_i32(BASE_POINT_RADIUS), BLUE.filled())
                }),
        )
        .map_err(|e| Error::msg(e.to_string()))?
        .label("данные")
        .legend(|(x, y)| {
            Circle::new((x, y), scaled_i32(BASE_LEGEND_POINT_RADIUS), BLUE.filled())
        });

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .label_font(("sans-serif", scaled_i32(BASE_SERIES_LABEL_FONT)))
        .draw()
        .map_err(|e| Error::msg(e.to_string()))?;

    root.present().map_err(|e| Error::msg(e.to_string()))?;

    Ok(())
}
