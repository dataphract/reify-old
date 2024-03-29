use std::{
    ffi::CStr,
    fmt::{self, Write},
};

use erupt::vk;

use crate::{vks, Instance};

fn format_cstr<F, C>(f: &mut F, cstr: C) -> fmt::Result
where
    F: fmt::Write,
    C: AsRef<CStr>,
{
    let mut start = 0;
    let bytes = cstr.as_ref().to_bytes();

    'format: while start < bytes.len() {
        let unvalidated = &bytes[start..];

        match std::str::from_utf8(unvalidated) {
            Ok(s) => {
                f.write_str(s)?;
                start += s.len();
            }

            Err(e) => {
                // Safety: validated up to `e.valid_up_to()`.
                let valid =
                    unsafe { std::str::from_utf8_unchecked(&unvalidated[..e.valid_up_to()]) };

                f.write_str(valid)?;
                f.write_char(char::REPLACEMENT_CHARACTER)?;
                match e.error_len() {
                    // Skip the validated substring and the unrecognized sequence.
                    Some(l) => start += valid.len() + l,

                    // Unexpected end of input.
                    None => break 'format,
                }
            }
        }
    }

    Ok(())
}

/// Format a list of `vk::DebugUtilsLabelEXT` objects.
///
/// # Safety
///
/// `labels` must be a pointer to a properly aligned sequence of `count`
/// `vk::DebugUtilsLabelEXT` objects.
unsafe fn format_debug_utils_label_ext<F>(
    f: &mut F,
    about: &str,
    labels: *mut vk::DebugUtilsLabelEXT,
    count: usize,
) -> fmt::Result
where
    F: fmt::Write,
{
    let labels = unsafe { std::slice::from_raw_parts(labels, count) };

    let (last, init) = match labels.split_last() {
        Some(li) => li,
        None => return Ok(()),
    };

    write!(f, "{}: ", about)?;

    for label in init {
        if let Some(label_ptr) = unsafe { label.p_label_name.as_ref() } {
            let label_cstr = unsafe { CStr::from_ptr(label_ptr) };
            format_cstr(f, label_cstr)?;
            f.write_str(", ")?;
        }
    }

    if let Some(label_ptr) = unsafe { last.p_label_name.as_ref() } {
        let label_cstr = unsafe { CStr::from_ptr(label_ptr) };
        format_cstr(f, label_cstr)?;
    }

    Ok(())
}

/// Format a list of `vk::DebugUtilsObjectNameInfoEXT` objects.
///
/// # Safety
///
/// `infos` must be a pointer to a properly aligned sequence of `count`
/// `vk::DebugUtilsObjectNameInfoEXT` objects.
unsafe fn format_debug_utils_object_name_info_ext<F>(
    f: &mut F,
    about: &str,
    infos: *mut vk::DebugUtilsObjectNameInfoEXT,
    count: usize,
) -> fmt::Result
where
    F: fmt::Write,
{
    let infos = unsafe { std::slice::from_raw_parts(infos, count) };

    let (last, init) = match infos.split_last() {
        Some(li) => li,
        None => return Ok(()),
    };

    for info in init {
        write!(
            f,
            "{}: (type: {:?}, handle: 0x{:X}",
            about, info.object_type, info.object_handle
        )?;
        if let Some(info_ptr) = unsafe { info.p_object_name.as_ref() } {
            let info_cstr = unsafe { CStr::from_ptr(info_ptr) };
            f.write_str("name: ")?;
            format_cstr(f, info_cstr)?;
        }
        f.write_str("), ")?;
    }

    write!(
        f,
        "{}: (type: {:?}, handle: 0x{:X}",
        about, last.object_type, last.object_handle
    )?;
    if let Some(info_ptr) = unsafe { last.p_object_name.as_ref() } {
        let info_cstr = unsafe { CStr::from_ptr(info_ptr) };
        f.write_str("name: ")?;
        format_cstr(f, info_cstr)?;
    }
    f.write_str(")")?;

    Ok(())
}

const DEBUG_MESSAGE_INIT_CAPACITY: usize = 128;
unsafe extern "system" fn debug_utils_messenger_callback(
    severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    // Via the spec: "The application should always return VK_FALSE."

    if std::thread::panicking() {
        return vk::FALSE;
    }

    if let Err(e) = debug_utils_messenger_callback_impl(severity, ty, callback_data, user_data) {
        log::error!("debug message formatting failed: {}", e);
    }

    vk::FALSE
}

fn debug_utils_messenger_callback_impl(
    severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> fmt::Result {
    let callback_data = unsafe { *callback_data };

    let severity = match severity {
        vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT => log::Level::Error,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT => log::Level::Warn,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT => log::Level::Info,
        vk::DebugUtilsMessageSeverityFlagBitsEXT::VERBOSE_EXT => log::Level::Trace,
        _ => log::Level::Warn,
    };

    let mut log_message = String::with_capacity(DEBUG_MESSAGE_INIT_CAPACITY);

    write!(&mut log_message, "{:?} ", ty)?;

    let msg_id_num = callback_data.message_id_number;
    if callback_data.p_message_id_name.is_null() {
        // "[Message MESSAGE_ID]"
        write!(&mut log_message, "[Message 0x{:X}] : ", msg_id_num)?;
    } else {
        // "[MESSAGE_NAME (MESSAGE_ID)]"
        let msg_name_cstr = unsafe { CStr::from_ptr(callback_data.p_message_id_name) };
        log_message.write_str("[")?;
        format_cstr(&mut log_message, msg_name_cstr)?;
        write!(&mut log_message, " (0x{:X})] : ", msg_id_num)?;
    };

    if !callback_data.p_message.is_null() {
        let msg_cstr = unsafe { CStr::from_ptr(callback_data.p_message) };
        format_cstr(&mut log_message, msg_cstr)?;
    };

    unsafe {
        format_debug_utils_label_ext(
            &mut log_message,
            "queue info",
            callback_data.p_queue_labels as *mut _,
            callback_data.queue_label_count as usize,
        )?;

        format_debug_utils_label_ext(
            &mut log_message,
            "cmdbuf info",
            callback_data.p_cmd_buf_labels as *mut _,
            callback_data.cmd_buf_label_count as usize,
        )?;

        format_debug_utils_object_name_info_ext(
            &mut log_message,
            "queue info",
            callback_data.p_queue_labels as *mut _,
            callback_data.queue_label_count as usize,
        )?;
    }

    log::log!(severity, "{}", log_message);

    Ok(())
}

pub struct DebugMessenger {
    instance: Instance,
    messenger: Option<vks::DebugUtilsMessengerEXT>,
}

impl Drop for DebugMessenger {
    fn drop(&mut self) {
        if let Some(messenger) = self.messenger.take() {
            unsafe {
                self.instance
                    .read_inner()
                    .handle()
                    .destroy_debug_utils_messenger(messenger);
            }
        }
    }
}

impl DebugMessenger {
    /// Initializes a new `DebugUtils`.
    pub fn new(instance: Instance) -> DebugMessenger {
        let debug_ext_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::empty())
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(debug_utils_messenger_callback));

        // Safety: messenger is destroyed in Drop impl.
        let messenger = unsafe {
            instance
                .read_inner()
                .handle()
                .create_debug_utils_messenger(&debug_ext_info)
        }
        .expect("failed to create debug messenger");

        DebugMessenger {
            instance,
            messenger: Some(messenger),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cstr() {
        // �
        let cases = &[
            (b"A valid C string\0".to_vec(), "A valid C string"),
            (b"Unexpected EOF: \xE0\0".to_vec(), "Unexpected EOF: �"),
            (
                b"Incomplete sequence: \xFA.\0".to_vec(),
                "Incomplete sequence: �.",
            ),
        ];

        let mut actual = String::new();
        for (input, expected) in cases {
            actual.clear();
            format_cstr(&mut actual, CStr::from_bytes_with_nul(input).unwrap()).unwrap();
            assert_eq!(expected, &actual);
        }
    }
}
