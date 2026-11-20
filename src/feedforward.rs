use std::io::{Read, Write};

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct FeedForward {
    w1: Matrix, b1: Vec<f64>, w2: Matrix, b2: Vec<f64>,
    grad_w1: Matrix, grad_b1: Vec<f64>, grad_w2: Matrix, grad_b2: Vec<f64>,
    embed_dim: usize, ff_dim: usize,
}

impl FeedForward {
    pub(crate) fn diagnostic_extrema(&self) -> Vec<(&'static str, f64, f64)> {
        vec![
            ("FFN_W1", max_abs(self.w1.iter().flatten().copied()), max_abs(self.grad_w1.iter().flatten().copied())),
            ("FFN_W2", max_abs(self.w2.iter().flatten().copied()), max_abs(self.grad_w2.iter().flatten().copied())),
        ]
    }
    pub fn new(embed_dim: usize, ff_dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let w1 = random_matrix(embed_dim, ff_dim, &mut rng);
        let w2 = random_matrix(ff_dim, embed_dim, &mut rng);
        Self { w1, b1: vec![0.0; ff_dim], w2, b2: vec![0.0; embed_dim], grad_w1: zeros(embed_dim, ff_dim), grad_b1: vec![0.0; ff_dim], grad_w2: zeros(ff_dim, embed_dim), grad_b2: vec![0.0; embed_dim], embed_dim, ff_dim }
    }
    pub fn forward(&self, x: &[Vec<f64>]) -> Matrix { self.forward_with_cache(x).0 }
    pub fn forward_with_cache(&self, x: &[Vec<f64>]) -> (Matrix, Matrix) {
        validate(x, self.embed_dim, "input");
        let mut hidden = zeros(x.len(), self.ff_dim);
        for (i, hidden_row) in hidden.iter_mut().enumerate() { for (j, value) in hidden_row.iter_mut().enumerate() { let z = self.b1[j] + (0..self.embed_dim).map(|k| x[i][k] * self.w1[k][j]).sum::<f64>(); *value = z.max(0.0); } }
        let mut output = zeros(x.len(), self.embed_dim);
        for (i, output_row) in output.iter_mut().enumerate() { for (j, value) in output_row.iter_mut().enumerate() { *value = self.b2[j] + (0..self.ff_dim).map(|k| hidden[i][k] * self.w2[k][j]).sum::<f64>(); } }
        (output, hidden)
    }
    pub fn backward(&mut self, x: &[Vec<f64>], hidden: &[Vec<f64>], d_output: &[Vec<f64>]) -> Matrix {
        validate(x, self.embed_dim, "input"); validate(hidden, self.ff_dim, "hidden"); validate(d_output, self.embed_dim, "output gradient");
        assert_eq!(x.len(), hidden.len(), "input and hidden sequence lengths must match"); assert_eq!(x.len(), d_output.len(), "input and output-gradient sequence lengths must match");
        let mut dx = zeros(x.len(), self.embed_dim);
        for pos in 0..x.len() {
            let mut dh = vec![0.0; self.ff_dim];
            for (j, &output_grad) in d_output[pos].iter().enumerate() { self.grad_b2[j] += output_grad; for (k, &hidden_value) in hidden[pos].iter().enumerate() { self.grad_w2[k][j] += hidden_value * output_grad; } }
            for k in 0..self.ff_dim {
                dh[k] = (0..self.embed_dim).map(|j| d_output[pos][j] * self.w2[k][j]).sum();
                if hidden[pos][k] <= 0.0 { dh[k] = 0.0; }
                self.grad_b1[k] += dh[k];
                for i in 0..self.embed_dim { self.grad_w1[i][k] += x[pos][i] * dh[k]; dx[pos][i] += dh[k] * self.w1[i][k]; }
            }
        }
        dx
    }
    pub fn apply_gradients(&mut self, learning_rate: f64) {
        for (w,g) in self.w1.iter_mut().flatten().zip(self.grad_w1.iter_mut().flatten()) { *w -= learning_rate * *g; *g=0.0; }
        for (w,g) in self.w2.iter_mut().flatten().zip(self.grad_w2.iter_mut().flatten()) { *w -= learning_rate * *g; *g=0.0; }
        for (w,g) in self.b1.iter_mut().zip(&mut self.grad_b1) { *w -= learning_rate * *g; *g=0.0; }
        for (w,g) in self.b2.iter_mut().zip(&mut self.grad_b2) { *w -= learning_rate * *g; *g=0.0; }
    }

    pub(crate) fn write_checkpoint_weights(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::checkpoint::write_matrix(writer, &self.w1)?;
        crate::checkpoint::write_f64s(writer, &self.b1)?;
        crate::checkpoint::write_matrix(writer, &self.w2)?;
        crate::checkpoint::write_f64s(writer, &self.b2)
    }

    pub(crate) fn read_checkpoint_weights(&mut self, reader: &mut dyn Read) -> Result<(), String> {
        self.w1 = crate::checkpoint::read_matrix(reader, self.embed_dim, self.ff_dim)
            .map_err(|e| e.to_string())?;
        self.b1 = crate::checkpoint::read_f64s(reader, self.ff_dim).map_err(|e| e.to_string())?;
        self.w2 = crate::checkpoint::read_matrix(reader, self.ff_dim, self.embed_dim)
            .map_err(|e| e.to_string())?;
        self.b2 = crate::checkpoint::read_f64s(reader, self.embed_dim).map_err(|e| e.to_string())?;
        self.grad_w1 = zeros(self.embed_dim, self.ff_dim);
        self.grad_b1 = vec![0.0; self.ff_dim];
        self.grad_w2 = zeros(self.ff_dim, self.embed_dim);
        self.grad_b2 = vec![0.0; self.embed_dim];
        Ok(())
    }
    pub fn check_gradients(embed_dim: usize, ff_dim: usize, seq_len: usize, seed: u64) -> (bool, f64) {
        if embed_dim == 0 || ff_dim == 0 || seq_len == 0 { return (false, f64::INFINITY); }
        let mut model = Self::new(embed_dim, ff_dim, seed); let mut rng = SplitMix64::new(seed.wrapping_add(991)); let x = random_matrix(seq_len, embed_dim, &mut rng);
        let (_, hidden) = model.forward_with_cache(&x); let d_out = vec![vec![1.0; embed_dim]; seq_len]; model.backward(&x, &hidden, &d_out); let analytic=model.grad_w1.clone(); let eps=1e-5; let mut max_error:f64=0.0;
        for idx in 0..8.min(embed_dim*ff_dim) { let r=idx/ff_dim; let c=idx%ff_dim; model.w1[r][c]+=eps; let plus=sum(&model.forward(&x)); model.w1[r][c]-=2.0*eps; let minus=sum(&model.forward(&x)); model.w1[r][c]+=eps; let numerical=(plus-minus)/(2.0*eps); let a=analytic[r][c]; let rel=(numerical-a).abs()/(1e-8+numerical.abs().max(a.abs())); max_error=max_error.max(rel); }
        (max_error < 1e-4, max_error)
    }

    #[cfg(test)]
    pub(crate) fn test_parameter_value_and_gradient(&self, kind: usize, row: usize, column: usize) -> (f64, f64) {
        match kind {
            0 => (self.w1[row][column], self.grad_w1[row][column]),
            1 => (self.w2[row][column], self.grad_w2[row][column]),
            2 => (self.b1[row], self.grad_b1[row]),
            3 => (self.b2[row], self.grad_b2[row]),
            _ => panic!("unknown FeedForward parameter kind"),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_perturb_parameter(&mut self, kind: usize, row: usize, column: usize, delta: f64) {
        match kind {
            0 => self.w1[row][column] += delta,
            1 => self.w2[row][column] += delta,
            2 => self.b1[row] += delta,
            3 => self.b2[row] += delta,
            _ => panic!("unknown FeedForward parameter kind"),
        }
    }
}
fn max_abs(values: impl Iterator<Item = f64>) -> f64 { values.map(f64::abs).fold(0.0, f64::max) }
fn validate(a: &[Vec<f64>], width: usize, name: &str) { assert!(a.iter().all(|r| r.len()==width), "{name} rows must have width {width}"); }
fn zeros(m:usize,n:usize)->Matrix { vec![vec![0.0;n];m] }
fn random_matrix(m:usize,n:usize,rng:&mut SplitMix64)->Matrix { (0..m).map(|_|(0..n).map(|_|(rng.next_f64()*2.0-1.0)*0.05).collect()).collect() }
fn sum(a:&[Vec<f64>])->f64 { a.iter().flatten().sum() }
struct SplitMix64(u64);
impl SplitMix64 { fn new(seed:u64)->Self{Self(seed)} fn next_f64(&mut self)->f64{self.0=self.0.wrapping_add(0x9E3779B97F4A7C15);let mut z=self.0;z=(z^(z>>30)).wrapping_mul(0xBF58476D1CE4E5B9);z=(z^(z>>27)).wrapping_mul(0x94D049BB133111EB);z^=z>>31;(z>>11)as f64/((1u64<<53)as f64)} }

#[cfg(test)]
mod tests {
 use super::*;
 #[test] fn shape_and_gradient_check(){let m=FeedForward::new(8,32,5);let x=vec![vec![0.2;8];4];let y=m.forward(&x);assert_eq!(y.len(),4);assert_eq!(y[0].len(),8);let(pass,error)=FeedForward::check_gradients(8,32,4,5);assert!(pass,"max relative error {error}");println!("Feed-forward W1 max relative gradient error: {error:.3e}");}
 #[test] fn relu_zeroes_gradient_for_inactive_units(){let mut m=FeedForward::new(2,2,1);m.w1=vec![vec![-1.0,1.0],vec![-1.0,1.0]];m.b1=vec![0.0,0.0];m.w2=vec![vec![1.0,1.0],vec![1.0,1.0]];let x=vec![vec![1.0,1.0]];let(_,h)=m.forward_with_cache(&x);assert_eq!(h[0],vec![0.0,2.0]);m.backward(&x,&h,&[vec![1.0,1.0]]);assert_eq!(m.grad_w2[0],vec![0.0,0.0]);assert_eq!(m.grad_b1[0],0.0);assert_eq!(m.grad_w1[0][0],0.0);}
}
