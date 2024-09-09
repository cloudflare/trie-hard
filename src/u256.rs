// Copyright 2024 Cloudflare, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    cmp::Ordering,
    ops::{
        Add, AddAssign, BitAnd, BitOrAssign, Shl, ShlAssign, Sub, SubAssign,
    },
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct U256([u64; 4]);

impl U256 {
    pub fn count_ones(&self) -> u32 {
        self.0.iter().cloned().map(u64::count_ones).sum()
    }
}

impl BitAnd for U256 {
    type Output = Self;
    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(l, r)| *l &= *r);
        self
    }
}

impl BitOrAssign for U256 {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(l, r)| *l |= r);
    }
}

impl PartialEq<u64> for U256 {
    fn eq(&self, other: &u64) -> bool {
        self.0[0] == *other && self.0.iter().skip(1).all(|p| *p == 0)
    }
}

impl PartialOrd<u64> for U256 {
    fn partial_cmp(&self, other: &u64) -> Option<Ordering> {
        if self.0.iter().skip(1).any(|p| *p > 0) {
            Some(Ordering::Greater)
        } else {
            Some(self.0[0].cmp(other))
        }
    }
}

impl AddAssign<u64> for U256 {
    fn add_assign(&mut self, rhs: u64) {
        let mut overflow = rhs;
        for p in self.0.iter_mut() {
            let (result, did_overflow) = p.overflowing_add(overflow);
            *p = result;
            overflow = did_overflow as u64;
        }
    }
}

impl Add<u64> for U256 {
    type Output = Self;
    fn add(mut self, rhs: u64) -> Self::Output {
        self.add_assign(rhs);
        self
    }
}

impl SubAssign<u64> for U256 {
    fn sub_assign(&mut self, rhs: u64) {
        let mut overflow = rhs;
        for p in self.0.iter_mut() {
            let (result, did_overflow) = p.overflowing_sub(overflow);
            *p = result;
            overflow = did_overflow as u64;
        }
    }
}

impl Sub<u64> for U256 {
    type Output = Self;
    fn sub(mut self, rhs: u64) -> Self::Output {
        self.sub_assign(rhs);
        self
    }
}

impl ShlAssign<u32> for U256 {
    fn shl_assign(&mut self, rhs: u32) {
        let carry_mask = 0xFFFFFFFF_FFFFFFFF << (64_u32.overflowing_sub(rhs).0);
        let mut carry = 0;

        for p in self.0.iter_mut() {
            let next_carry = (*p & carry_mask) >> (64 - rhs);
            *p = *p << rhs | carry;
            carry = next_carry;
        }
    }
}

impl Shl<u32> for U256 {
    type Output = Self;

    fn shl(mut self, rhs: u32) -> Self::Output {
        self.shl_assign(rhs);
        self
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_and() {
        let left = U256([0b0101010, 0b11100111, 0b111111, 0b00000]);
        let right = U256([0b0011100, 0b10100101, 0b100001, 0b01110]);
        let and = U256([0b0001000, 0b10100101, 0b100001, 0b00000]);

        assert_eq!(left & right, and);
    }

    #[test]
    fn test_add() {
        let left = U256([
            0xFFFFFFFF_FFFFFFFF,
            0xFFFFFFFF_FFFFFFFF,
            0xFFFFFFFF_FFFFFFFF,
            0xEFFFFFFF_FFFFFFFF,
        ]);
        let sum = U256([0, 0, 0, 0xF0000000_00000000]);

        assert_eq!(sum, left + 1);
    }

    #[test]
    fn test_sub() {
        let result = U256([
            0xFFFFFFFF_FFFFFFFF,
            0xFFFFFFFF_FFFFFFFF,
            0xFFFFFFFF_FFFFFFFF,
            0xEFFFFFFF_FFFFFFFF,
        ]);
        let left = U256([0, 0, 0, 0xF0000000_00000000]);

        assert_eq!(result, left - 1);
    }

    #[test]
    fn test_shl() {
        let left = U256([
            0b0101010111001110010101011100111001010101110011100101010111001110,
            0b1101010111001110010101011100111001010101110011100101010111001110,
            0b0101010111001110010101011100111001010101110011100101010111001110,
            0b1101010111001110010101011100111001010101110011100101010111001110,
        ]);
        let shl = U256([
            0b1010101110011100101010111001110010101011100111001010101110011100,
            0b1010101110011100101010111001110010101011100111001010101110011100,
            0b1010101110011100101010111001110010101011100111001010101110011101,
            0b1010101110011100101010111001110010101011100111001010101110011100,
        ]);

        assert_eq!(left << 1, shl);
    }

    #[test]
    fn test_ord() {
        assert_eq!(Some(Ordering::Equal), U256([1, 0, 0, 0]).partial_cmp(&1));
        assert_eq!(Some(Ordering::Less), U256([0, 0, 0, 0]).partial_cmp(&1));
        assert_eq!(Some(Ordering::Greater), U256([2, 0, 0, 0]).partial_cmp(&1));
        assert_eq!(Some(Ordering::Greater), U256([1, 0, 1, 0]).partial_cmp(&1));
        assert_eq!(Some(Ordering::Greater), U256([0, 1, 0, 0]).partial_cmp(&1));
    }
}
