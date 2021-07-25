use std::{cmp, collections::HashSet, fmt, hash::Hash, iter::FromIterator, mem};

use arrayvec::ArrayVec;

/// A dummy type which emits an error message when dropped.
///
/// This is useful for drawing attention to incorrect use of types which require
/// explicit destruction.
#[derive(Default)]
pub struct ErrorOnDrop<T>
where
    T: fmt::Display + Default,
{
    error: T,
    armed: bool,
}

impl<T> Drop for ErrorOnDrop<T>
where
    T: fmt::Display + Default,
{
    fn drop(&mut self) {
        if self.armed {
            log::error!("Error on drop: {}", self.error);
        }
    }
}

impl<T> ErrorOnDrop<T>
where
    T: fmt::Display + Default,
{
    pub fn new(error: T) -> ErrorOnDrop<T> {
        ErrorOnDrop { error, armed: true }
    }

    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

pub enum SmallSet<T: PartialEq, const CAP: usize> {
    Inline(ArrayVec<T, CAP>),
    Heap(HashSet<T>),
}

impl<T: PartialEq, const CAP: usize> Default for SmallSet<T, CAP> {
    fn default() -> Self {
        SmallSet::Inline(ArrayVec::new())
    }
}

impl<T, const CAP: usize> SmallSet<T, CAP>
where
    T: Eq + Hash,
{
    pub fn new() -> SmallSet<T, CAP> {
        SmallSet::Inline(ArrayVec::new())
    }

    pub fn capacity(&self) -> usize {
        match self {
            SmallSet::Inline(_) => CAP,
            SmallSet::Heap(s) => s.capacity(),
        }
    }

    pub fn is_inline(&self) -> bool {
        matches!(self, SmallSet::Inline(_))
    }

    pub fn is_heap(&self) -> bool {
        matches!(self, SmallSet::Heap(_))
    }

    pub fn reserve(&mut self, additional: usize) {
        match self {
            SmallSet::Inline(v) => {
                if v.len() + additional > CAP {
                    let s = Self::make_hashset(mem::take(v), CAP + additional);
                    *self = SmallSet::Heap(s);
                }
            }
            SmallSet::Heap(s) => s.reserve(additional),
        }
    }

    fn make_hashset(v: ArrayVec<T, CAP>, min_cap: usize) -> HashSet<T> {
        let cap = cmp::max(CAP, min_cap);
        let mut s = HashSet::with_capacity(cap);
        s.extend(v);
        s
    }

    fn move_to_heap(&mut self) {
        if let SmallSet::Inline(v) = self {
            let v = std::mem::take(v);
            let set = HashSet::from_iter(v.into_iter());
            *self = SmallSet::Heap(set);
        }
    }

    pub fn contains(&self, item: &T) -> bool {
        match self {
            SmallSet::Inline(s) => s.contains(item),
            SmallSet::Heap(s) => s.contains(item),
        }
    }

    pub fn insert(&mut self, item: T) -> bool {
        match self {
            SmallSet::Inline(ref mut s) => {
                if s.contains(&item) {
                    return true;
                }

                if let Err(item) = s.try_push(item) {
                    let mut set = Self::make_hashset(mem::take(s), CAP * 2);
                    set.insert(item.element());
                    *self = SmallSet::Heap(set);
                }

                false
            }
            SmallSet::Heap(s) => s.insert(item),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        match self {
            SmallSet::Inline(s) => SmallSetIter::Inline(s.iter()),
            SmallSet::Heap(s) => SmallSetIter::Heap(s.iter()),
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        match self {
            SmallSet::Inline(s) => s.retain(|x| f(x)),
            SmallSet::Heap(s) => s.retain(f),
        }
    }

    // TODO: extend()
}

pub enum SmallSetIter<'a, T> {
    Inline(std::slice::Iter<'a, T>),
    Heap(std::collections::hash_set::Iter<'a, T>),
}

impl<'a, T> Iterator for SmallSetIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SmallSetIter::Inline(it) => it.next(),
            SmallSetIter::Heap(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            SmallSetIter::Inline(it) => it.size_hint(),
            SmallSetIter::Heap(it) => it.size_hint(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_cap_is_array_cap() {
        let set = SmallSet::<(), 8>::new();
        assert_eq!(set.capacity(), 8);
    }

    #[test]
    fn contains() {
        let mut set = SmallSet::<u32, 2>::new();
        assert!(!set.contains(&42));
        set.insert(42);
        assert!(set.contains(&42));
    }
}
