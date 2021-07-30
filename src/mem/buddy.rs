use std::{cmp, collections::LinkedList, convert::TryInto, fmt};

use fixedbitset::FixedBitSet;

fn round_up_pow2(x: u64) -> Option<u64> {
    match x {
        0 => None,
        1 => Some(1),
        x if x >= (1 << 63) => None,
        _ => Some(2u64.pow((x - 1).log2() as u32 + 1)),
    }
}

fn round_down_pow2(x: u64) -> Option<u64> {
    match x {
        0 => None,
        1 => Some(1),
        _ => Some(2u64.pow(x.log2() as u32)),
    }
}

#[derive(Debug)]
pub struct BuddyError {
    msg: String,
}

impl fmt::Display for BuddyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for BuddyError {}

#[derive(Debug)]
struct BuddyLevel {
    block_size: u64,

    // A list of free blocks.
    free_list: LinkedList<BuddyBlock>,

    // One bit per pair of blocks.
    //
    // Given a pair of blocks (A, B) represented by bit X:
    //     X == is_free(A) XOR is_free(B)
    buddies: FixedBitSet,

    // One bit per block.
    //
    // If the bit is set, the block has been split.
    //
    // This field should be `None` for the leaf level.
    splits: Option<FixedBitSet>,
}

impl BuddyLevel {
    fn new(capacity: u64, block_size: u64, is_leaf: bool) -> BuddyLevel {
        assert!(capacity > 0);
        assert_eq!(block_size.count_ones(), 1);
        assert_eq!(capacity % block_size, 0);

        let num_blocks: usize = (capacity / block_size).try_into().unwrap();

        BuddyLevel {
            block_size,
            free_list: LinkedList::new(),
            buddies: FixedBitSet::with_capacity(num_blocks / 2),
            splits: if is_leaf {
                None
            } else {
                Some(FixedBitSet::with_capacity(num_blocks))
            },
        }
    }

    #[inline]
    fn index_of(&self, addr: u64) -> usize {
        assert_eq!(addr % self.block_size, 0);

        (addr / self.block_size).try_into().unwrap()
    }

    #[inline]
    fn buddy_bit(&self, addr: u64) -> usize {
        assert_eq!(addr % self.block_size, 0);

        self.index_of(addr) / 2
    }

    #[inline]
    fn buddy_addr(&self, addr: u64) -> u64 {
        assert_eq!(addr % self.block_size, 0);

        addr ^ self.block_size
    }

    /// Allocates a block from this level and toggles the corresponding buddy bit.
    fn allocate_one(&mut self) -> Option<BuddyBlock> {
        let block = self.free_list.pop_front()?;
        self.buddies.toggle(self.buddy_bit(block.addr));
        Some(block)
    }

    /// Assigns half a block from the previous level to this level.
    fn assign_half(&mut self, block: BuddyBlock) {
        let buddy_bit = self.buddy_bit(block.addr);
        assert!(!self.buddies.contains(buddy_bit));
        self.buddies.set(buddy_bit, true);
        self.free_list.push_front(block);
    }

    fn free(&mut self, block: BuddyBlock, coalesce: bool) -> Option<BuddyBlock> {
        let buddy_bit = self.buddy_bit(block.addr);
        self.buddies.toggle(buddy_bit);

        let split_bit = self.index_of(block.addr);
        if let Some(splits) = self.splits.as_mut() {
            splits.set(split_bit, false);
        }

        if !coalesce || self.buddies.contains(buddy_bit) {
            self.free_list.push_front(block);
            None
        } else {
            let buddy_addr = self.buddy_addr(block.addr);

            // Remove the buddy block from the free list.
            if self
                .free_list
                .drain_filter(|elem| elem.addr == buddy_addr)
                .next()
                .is_none()
            {
                panic!("missing buddy in free list");
            }

            // Return the coalesced block.
            Some(BuddyBlock {
                addr: block.addr & !self.block_size,
            })
        }
    }
}

#[derive(Debug)]
pub struct BuddyBlock {
    addr: u64,
}

#[derive(Debug)]
pub struct BuddyAllocator {
    capacity: u64,
    max_block_size: u64,
    min_block_size: u64,

    levels: Vec<BuddyLevel>,
}

impl BuddyAllocator {
    /// Calculates the level from which a block should be allocated.
    fn alloc_level(&self, size: u64) -> Result<usize, BuddyError> {
        if size > self.max_block_size {
            return Err(BuddyError {
                msg: format!("allocation size ({}) too large", size),
            });
        }

        let alloc_size = cmp::max(round_up_pow2(size).unwrap(), self.min_block_size);
        let level = (self.max_block_size.log2() - alloc_size.log2())
            .try_into()
            .unwrap();

        Ok(level)
    }

    /// Calculates the minimum level at which a block may be freed.
    ///
    /// This does not indicate the level at which the block *should* be freed,
    /// but rather uses the block address to eliminate levels whose block sizes
    /// are too large.
    fn min_free_level(&self, addr: u64) -> usize {
        if addr == 0 {
            return 0;
        }

        // The maximum possible size of the block.
        let max_size = 1 << addr.trailing_zeros();

        if max_size > self.max_block_size {
            return 0;
        }

        assert!(max_size >= self.min_block_size);

        (self.max_block_size.log2() - max_size.log2())
            .try_into()
            .unwrap()
    }

    pub fn allocate(&mut self, size: u64) -> Result<BuddyBlock, BuddyError> {
        if size == 0 {
            return Err(BuddyError {
                msg: "cannot allocate 0 bytes".into(),
            });
        }

        let target_level = self.alloc_level(size)?;

        // If there is a free block of the correct size, return it immediately.
        if let Some(block) = self.levels[target_level].allocate_one() {
            return Ok(block);
        }

        // Otherwise, scan increasing block sizes until a free block is found.
        let (block, init_level) = (0..target_level)
            .rev()
            .find_map(|level| self.levels[level].allocate_one().map(|blk| (blk, level)))
            .ok_or_else(|| BuddyError {
                msg: "out of memory".into(),
            })?;

        // Once a free block is found, split it repeatedly to obtain a
        // suitably sized block.
        for level in init_level..target_level {
            // Split the block. The address of the front half does not change.
            let back_half = BuddyBlock {
                addr: block.addr + self.levels[level].block_size / 2,
            };

            // Mark the block as split.
            let split_bit = self.levels[level].index_of(block.addr);
            if let Some(s) = self.levels[level].splits.as_mut() {
                s.set(split_bit, true);
            }

            // Add one half of the split block to the next level's free list.
            self.levels[level + 1].assign_half(back_half);
        }

        Ok(block)
    }

    pub fn free(&mut self, block: BuddyBlock) {
        // Some addresses can't come from earlier levels because their addresses
        // imply a smaller block size.
        let min_level = self.min_free_level(block.addr);

        let mut at_level = None;
        for level in min_level..self.levels.len() {
            if self.levels[level]
                .splits
                .as_ref()
                .map(|s| !s.contains(self.levels[level].index_of(block.addr)))
                .unwrap_or(true)
            {
                at_level = Some(level);
                break;
            }
        }

        let at_level = at_level.expect("no level found to free block");

        let mut block = Some(block);
        for level in (0..=at_level).rev() {
            match block.take() {
                Some(b) => {
                    block = self.levels[level].free(b, level != 0);
                }
                None => break,
            }
        }

        assert!(block.is_none(), "top level coalesced a block");
    }
}

// 128 MiB.
const DEFAULT_CAPACITY: u64 = 128 * 1024 * 1024;
// 4 MiB.
const DEFAULT_MAX_BLOCK_SIZE: u64 = 4 * 1024 * 1024;
// 4 KiB.
const DEFAULT_MIN_BLOCK_SIZE: u64 = 4 * 1024;

pub struct BuddyBuilder {
    capacity: u64,
    max_block_size: u64,
    min_block_size: u64,
}

impl Default for BuddyBuilder {
    fn default() -> Self {
        BuddyBuilder {
            capacity: DEFAULT_CAPACITY,
            max_block_size: DEFAULT_MAX_BLOCK_SIZE,
            min_block_size: DEFAULT_MIN_BLOCK_SIZE,
        }
    }
}

impl BuddyBuilder {
    pub fn new() -> BuddyBuilder {
        Self::default()
    }

    pub fn capacity(mut self, capacity: u64) -> BuddyBuilder {
        BuddyBuilder { capacity, ..self }
    }

    pub fn max_block_size(mut self, max_block_size: u64) -> BuddyBuilder {
        BuddyBuilder {
            max_block_size,
            ..self
        }
    }

    pub fn min_block_size(mut self, min_block_size: u64) -> BuddyBuilder {
        BuddyBuilder {
            min_block_size,
            ..self
        }
    }

    pub fn build(self) -> Result<BuddyAllocator, BuddyError> {
        if self.capacity == 0 {
            return Err(BuddyError {
                msg: "capacity must not be zero".into(),
            });
        }

        if self.max_block_size.count_ones() != 1 {
            return Err(BuddyError {
                msg: format!(
                    "maximum block size ({}) must be a power of two",
                    self.max_block_size
                ),
            });
        }

        if self.min_block_size.count_ones() != 1 {
            return Err(BuddyError {
                msg: format!(
                    "minimum block size ({}) must be a power of two",
                    self.min_block_size
                ),
            });
        }

        if self.min_block_size > self.max_block_size {
            return Err(BuddyError {
                msg: format!(
                    "minimum block size ({}) must not be greater than maximum block size ({})",
                    self.min_block_size, self.max_block_size
                ),
            });
        }

        if self.capacity % self.min_block_size != 0 {
            return Err(BuddyError {
                msg: format!(
                    "minimum block size ({}) must evenly divide capacity ({})",
                    self.min_block_size, self.capacity
                ),
            });
        }

        let cap_ceil = round_up_pow2(self.capacity).unwrap();
        let num_levels = self.max_block_size.log2() - self.min_block_size.log2() + 1;
        let mut levels = Vec::with_capacity(num_levels as usize);
        for log2 in 0..num_levels {
            let block_size = self.max_block_size / 2u64.pow(log2 as u32);
            levels.push(BuddyLevel::new(
                cap_ceil,
                block_size,
                log2 == num_levels - 1,
            ));
        }

        assert_eq!(
            levels.last().as_ref().unwrap().block_size,
            self.min_block_size
        );

        let mut alloc = BuddyAllocator {
            capacity: self.capacity,
            max_block_size: self.max_block_size,
            min_block_size: self.min_block_size,
            levels,
        };

        // Add blocks until the allocator has the correct capacity.
        let mut end = 0;
        while end < self.capacity {
            let remaining = self.capacity - end;
            let block_size = cmp::min(round_down_pow2(remaining).unwrap(), self.max_block_size);
            let level: usize = (self.max_block_size.log2() - block_size.log2())
                .try_into()
                .unwrap();
            let block = BuddyBlock { addr: end };
            let buddy_bit = alloc.levels[level].buddy_bit(block.addr);
            alloc.levels[level].buddies.toggle(buddy_bit);
            alloc.levels[level].free_list.push_back(block);
            end += block_size;
        }

        Ok(alloc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_capacity() {
        assert!(BuddyBuilder::new().capacity(0).build().is_err());
    }

    #[test]
    fn max_size_not_power_of_two() {
        assert!(BuddyBuilder::new().max_block_size(777).build().is_err());
    }

    #[test]
    fn min_size_not_power_of_two() {
        assert!(BuddyBuilder::new().min_block_size(3).build().is_err());
    }

    #[test]
    fn min_size_gt_max_size() {
        assert!(BuddyBuilder::new()
            .capacity(128)
            .min_block_size(16)
            .max_block_size(4)
            .build()
            .is_err());
    }

    #[test]
    fn alloc_zero() {
        let mut buddy = BuddyBuilder::new().build().unwrap();

        assert!(buddy.allocate(0).is_err());
    }

    #[test]
    fn alloc_too_large() {
        let mut buddy = BuddyBuilder::new()
            .capacity(4096)
            .max_block_size(1024)
            .min_block_size(8)
            .build()
            .unwrap();

        assert!(buddy.allocate(1025).is_err());
    }

    #[test]
    fn alloc_oom() {
        let mut buddy = BuddyBuilder::new()
            .capacity(112)
            .max_block_size(64)
            .min_block_size(16)
            .build()
            .unwrap();

        buddy.allocate(64).unwrap();
        buddy.allocate(16).unwrap();
        buddy.allocate(32).unwrap();
        assert!(buddy.allocate(1).is_err());
    }

    #[test]
    fn capacity_not_power_of_two() {
        let buddy = BuddyBuilder::new()
            .capacity(248)
            .max_block_size(128)
            .min_block_size(8)
            .build()
            .unwrap();

        // Should look like:
        // 0                              128             192     224  240 248
        // |                               |               |       |     | |
        // [              128              ]               |       |     | |
        //                                 [       64      ]       |     | |
        //                                                 [   32  ]     | |
        //                                                         [  16 ] |
        //                                                               [8]
        for level in buddy.levels.iter() {
            assert_eq!(level.free_list.len(), 1);
        }
    }

    #[test]
    fn alloc_one() {
        let mut buddy = BuddyBuilder::new()
            .capacity(512)
            .max_block_size(128)
            .min_block_size(8)
            .build()
            .unwrap();

        assert_eq!(buddy.levels[0].free_list.len(), 4);

        buddy.allocate(8);

        assert_eq!(buddy.levels[0].free_list.len(), 3);
        for level in 1..buddy.levels.len() {
            assert_eq!(buddy.levels[level].free_list.len(), 1);
        }
    }

    #[test]
    fn alloc_forward_free_forward() {
        let mut buddy = BuddyBuilder::new()
            .capacity(256)
            .max_block_size(64)
            .min_block_size(8)
            .build()
            .unwrap();

        let mut blocks = Vec::new();

        for i in 0..=6 {
            let block = buddy.allocate(2u64.pow(i)).unwrap();
            blocks.push(block);
        }

        for block in blocks.drain(..) {
            buddy.free(block);
        }

        assert_eq!(buddy.levels[0].free_list.len(), 4);
        assert_eq!(buddy.levels[0].buddies.count_ones(..), 0);
        assert_eq!(buddy.levels[0].splits.as_ref().unwrap().count_ones(..), 0);
        for level in &buddy.levels[1..] {
            assert_eq!(level.free_list.len(), 0);
            assert_eq!(level.buddies.count_ones(..), 0);
            if let Some(s) = level.splits.as_ref() {
                assert_eq!(s.count_ones(..), 0);
            }
        }
    }

    #[test]
    fn alloc_forward_free_backward() {
        let mut buddy = BuddyBuilder::new()
            .capacity(256)
            .max_block_size(64)
            .min_block_size(8)
            .build()
            .unwrap();

        let mut blocks = Vec::new();

        for i in 0..=6 {
            let block = buddy.allocate(2u64.pow(i)).unwrap();
            blocks.push(block);
        }

        for block in blocks.drain(..).rev() {
            buddy.free(block);
        }

        assert_eq!(buddy.levels[0].free_list.len(), 4);
        assert_eq!(buddy.levels[0].buddies.count_ones(..), 0);
        assert_eq!(buddy.levels[0].splits.as_ref().unwrap().count_ones(..), 0);
        for level in &buddy.levels[1..] {
            assert_eq!(level.free_list.len(), 0);
            assert_eq!(level.buddies.count_ones(..), 0);
            if let Some(s) = level.splits.as_ref() {
                assert_eq!(s.count_ones(..), 0);
            }
        }
    }
}
