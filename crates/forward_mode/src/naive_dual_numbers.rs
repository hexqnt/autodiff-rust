use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct NaiveDual {
    value: f64,
    derivatives: f64,
}

impl NaiveDual {
    #[must_use]
    pub const fn constant(value: f64) -> Self {
        Self {
            value,
            derivatives: 0.0,
        }
    }

    #[must_use]
    pub const fn variable(value: f64) -> Self {
        Self {
            value,
            derivatives: 1.0,
        }
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
            derivatives: self.derivatives * derivative_factor,
        }
    }

    #[must_use]
    pub fn cos(self) -> Self {
        let value = self.value.cos();
        let derivative_factor = -self.value.sin();
        Self {
            value,
            derivatives: self.derivatives * derivative_factor,
        }
    }

    #[must_use]
    pub fn tan(self) -> Self {
        let value = self.value.tan();
        let cos = self.value.cos();
        let derivative_factor = 1.0 / (cos * cos);
        Self {
            value,
            derivatives: self.derivatives * derivative_factor,
        }
    }

    #[must_use]
    pub fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            derivatives: self.derivatives * value,
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
            derivatives: self.derivatives / self.value,
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
            derivatives: self.derivatives * derivative_factor,
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
            derivatives: self.derivatives * derivative_factor,
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
            derivatives: self.derivatives * derivative_factor,
        }
    }

    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    #[must_use]
    pub const fn derivative(&self) -> f64 {
        self.derivatives
    }
}

impl From<f64> for NaiveDual {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Add for NaiveDual {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            derivatives: self.derivatives + rhs.derivatives,
        }
    }
}

impl Add<f64> for NaiveDual {
    type Output = Self;

    fn add(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            derivatives: self.derivatives,
        }
    }
}

impl Sub for NaiveDual {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            derivatives: self.derivatives - rhs.derivatives,
        }
    }
}

impl Sub<f64> for NaiveDual {
    type Output = Self;

    fn sub(self, rhs: f64) -> Self {
        Self {
            value: self.value - rhs,
            derivatives: self.derivatives,
        }
    }
}

impl Mul for NaiveDual {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            derivatives: self.value * rhs.derivatives + rhs.value * self.derivatives,
        }
    }
}

impl Mul<f64> for NaiveDual {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            value: self.value * rhs,
            derivatives: self.derivatives * rhs,
        }
    }
}

impl Div for NaiveDual {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.value * rhs.value;
        Self {
            value: self.value / rhs.value,
            derivatives: (self.derivatives * rhs.value - self.value * rhs.derivatives)
                / denominator,
        }
    }
}

impl Div<f64> for NaiveDual {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            value: self.value / rhs,
            derivatives: self.derivatives / rhs,
        }
    }
}

impl Neg for NaiveDual {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            value: -self.value,
            derivatives: -self.derivatives,
        }
    }
}
