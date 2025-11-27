use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy)]
pub struct Dual<const N: usize> {
    value: f64,
    derivatives: [f64; N],
}

#[must_use]
pub fn variables<const N: usize>(values: [f64; N]) -> [Dual<N>; N] {
    std::array::from_fn(|index| Dual::variable(index, values[index]))
}

impl<const N: usize> fmt::Debug for Dual<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Dual")
            .field("value", &self.value)
            .field("derivatives", &self.derivatives)
            .finish()
    }
}

impl<const N: usize> Dual<N> {
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            derivatives: [0.0; N],
        }
    }

    #[must_use]
    pub fn variable(index: usize, value: f64) -> Self {
        assert!(
            index < N,
            "Variable index {index} is out of range for {N} variables."
        );
        let mut derivatives = [0.0; N];
        derivatives[index] = 1.0;
        Self { value, derivatives }
    }

    #[must_use]
    pub const fn indicator(condition: bool) -> Self {
        if condition {
            Self::constant(1.0)
        } else {
            Self::constant(0.0)
        }
    }

    #[must_use]
    pub fn sin(self) -> Self {
        let value = self.value.sin();
        let derivative_factor = self.value.cos();
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub fn cos(self) -> Self {
        let value = self.value.cos();
        let derivative_factor = -self.value.sin();
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub fn tan(self) -> Self {
        let value = self.value.tan();
        let cos = self.value.cos();
        let derivative_factor = 1.0 / (cos * cos);
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            derivatives: self.derivatives.map(|derivative| derivative * value),
        }
    }

    #[must_use]
    pub fn ln(self) -> Self {
        assert!(
            self.value > 0.0,
            "ln is only defined for positive values, received {}",
            self.value
        );
        Self {
            value: self.value.ln(),
            derivatives: self.derivatives.map(|derivative| derivative / self.value),
        }
    }

    #[must_use]
    pub fn sqrt(self) -> Self {
        assert!(
            self.value >= 0.0,
            "sqrt is only defined for non-negative values, received {}",
            self.value
        );
        let value = self.value.sqrt();
        let derivative_factor = 0.5 / value;
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub fn powi(self, exponent: i32) -> Self {
        let value = self.value.powi(exponent);
        let derivative_factor = if exponent == 0 {
            0.0
        } else {
            f64::from(exponent) * self.value.powi(exponent - 1)
        };
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub fn powf(self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        let derivative_factor = if exponent == 0.0 {
            0.0
        } else {
            exponent * self.value.powf(exponent - 1.0)
        };
        Self {
            value,
            derivatives: self
                .derivatives
                .map(|derivative| derivative * derivative_factor),
        }
    }

    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    #[must_use]
    pub const fn derivatives(&self) -> &[f64; N] {
        &self.derivatives
    }
}

impl<const N: usize> From<f64> for Dual<N> {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl<const N: usize> Add for Dual<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut derivatives = [0.0; N];
        for i in 0..N {
            derivatives[i] = self.derivatives[i] + rhs.derivatives[i];
        }
        Self {
            value: self.value + rhs.value,
            derivatives,
        }
    }
}

impl<const N: usize> Add<f64> for Dual<N> {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            derivatives: self.derivatives,
        }
    }
}

impl<const N: usize> Sub for Dual<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut derivatives = [0.0; N];
        for i in 0..N {
            derivatives[i] = self.derivatives[i] - rhs.derivatives[i];
        }
        Self {
            value: self.value - rhs.value,
            derivatives,
        }
    }
}

impl<const N: usize> Sub<f64> for Dual<N> {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            derivatives: self.derivatives,
        }
    }
}

impl<const N: usize> Mul for Dual<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut derivatives = [0.0; N];
        for i in 0..N {
            derivatives[i] = self.value * rhs.derivatives[i] + rhs.value * self.derivatives[i];
        }
        Self {
            value: self.value * rhs.value,
            derivatives,
        }
    }
}

impl<const N: usize> Mul<f64> for Dual<N> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            value: self.value * rhs,
            derivatives: self.derivatives.map(|derivative| derivative * rhs),
        }
    }
}

impl<const N: usize> Div for Dual<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.value * rhs.value;
        let mut derivatives = [0.0; N];
        for i in 0..N {
            derivatives[i] =
                (self.derivatives[i] * rhs.value - self.value * rhs.derivatives[i]) / denominator;
        }
        Self {
            value: self.value / rhs.value,
            derivatives,
        }
    }
}

impl<const N: usize> Div<f64> for Dual<N> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            value: self.value / rhs,
            derivatives: self.derivatives.map(|derivative| derivative / rhs),
        }
    }
}

impl<const N: usize> Neg for Dual<N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivatives: self.derivatives.map(|derivative| -derivative),
        }
    }
}
